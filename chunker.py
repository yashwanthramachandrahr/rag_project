import subprocess
import sys
import os
import json
import glob
import pandas as pd
import gc
from pypdf import PdfReader

def check_dependencies():
    """Checks if docling and docling-core are installed, and installs them if missing."""
    try:
        from docling.document_converter import DocumentConverter
        from docling_core.types.doc.document import TableItem
        print("✅ Dependencies found.")
    except ImportError:
        print("🔍 Dependency missing. Attempting to install 'docling', 'docling-core', and 'pypdfium2'...")
        try:
            # Install docling and docling-core
            subprocess.check_call([sys.executable, "-m", "pip", "install", "docling", "docling-core", "pypdfium2"])
            print("🚀 Dependencies installed successfully. Restarting imports...")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please run: pip install docling docling-core")
            sys.exit(1)

# Check dependencies before proceeding with heavy imports
check_dependencies()

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HierarchicalChunker
from docling_core.types.doc.document import TableItem, TextItem, SectionHeaderItem, ListItem
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
 
# -------- Configuration --------
CHUNKS_DIR = "chunks"

# Ensure chunks directory exists
if not os.path.exists(CHUNKS_DIR):
    os.makedirs(CHUNKS_DIR)
 
# -------- Configuration --------
# No hardcoded PDF_PATH anymore
# OUTPUT_JSON pattern will be derived from PDF filename
 
# -------- Helper Functions --------
def get_context_string(headings):
    """Returns the last 2 headings as context, joined by ' > '."""
    if not headings:
        return ""
    # Take up to the last 2 headings
    return " > ".join(headings[-2:])
 
def process_table_item(item, doc, context):
    """
    Custom table chunking: Explodes a TableItem into row-by-row chunks
    with 'Columns: ... | Values: ...' format for LLM-friendly consumption.
    Ref: hybrid_docling_chunking.py
    """
    new_chunks = []
    try:
        df = item.export_to_dataframe(doc)
       
        # Universal Forward Fill for Merged Cells
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.ffill()
        df.fillna("", inplace=True)
 
        columns = df.columns.tolist()
        schema_str = " | ".join([str(c).strip() for c in columns])
       
        for index, row in df.iterrows():
            values_clean = []
            for val in row:
                val_str = str(val).strip()
                values_clean.append(val_str if val_str else "")
           
            # Check if row has any content (skip completely empty rows)
            if any(values_clean):
                row_values_str = " | ".join(values_clean)
                row_text = f"Columns: {schema_str}\nValues: {row_values_str}"
                new_chunks.append({"text": row_text, "context": context})
               
    except Exception as e:
        print(f"Error processing table: {e}")
        pass
       
    return new_chunks
 
def get_pages_to_skip(pdf_name):
    """Interactively asks the user for page numbers to skip."""
    print(f"\n📄 Found PDF: {pdf_name}")
    print("👉 Enter page numbers to SKIP (comma/space separated, e.g., '1 2 5' or '1,2,5').")
    print("   Press ENTER to skip NOTHING.")
   
    user_input = input("   Pages to skip: ").strip()
   
    pages_to_skip = set()
    if user_input:
        # Split by comma or space
        parts = user_input.replace(',', ' ').split()
        for p in parts:
            try:
                # Page numbers are 1-based
                val = int(p)
                pages_to_skip.add(val)
            except ValueError:
                print(f"   ⚠️ Warning: '{p}' is not a valid number. Ignoring.")
   
    if pages_to_skip:
        print(f"   🚫 Skipping pages: {sorted(pages_to_skip)}")
    else:
        print("   ✅ Processing ALL pages.")
       
    return pages_to_skip
 
# -------- Main Execution --------
def main():
    # Find all PDFs in 'pdf' directory
    all_pdfs = glob.glob("*.pdf") + glob.glob("*.PDF")
    # Deduplicate in case file system is case insensitive but glob returned both
    all_pdfs = sorted(list(set(all_pdfs)))
   
    if not all_pdfs:
        print("❌ No PDF files found in the current directory.")
        return
 
    print(f"🔎 Found {len(all_pdfs)} PDF(s) to process.")
 
    for pdf_path in all_pdfs:
        print(f"\n{'='*60}")
       
        # 1. Setup Converter (Fresh instance per PDF to avoid memory leaks)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        # 2a. Ask user for skip pages
        skip_pages = get_pages_to_skip(pdf_path)
        
        page_range = None
        if skip_pages:
            sorted_skips = sorted(list(skip_pages))
            # Find the first non-skipped page (1-based)
            first_non_skipped = 1
            for p in sorted_skips:
                if p == first_non_skipped:
                    first_non_skipped += 1
                else:
                    break
            
            # Use pypdf to get total pages for a valid end range
            try:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                if first_non_skipped > 1:
                    if first_non_skipped <= total_pages:
                        print(f"   ⏭️ Skipping conversion of first {first_non_skipped - 1} pages...")
                        # PageRange is (start_index_1_based, end_index_1_based) in this Docling version
                        page_range = (first_non_skipped, total_pages)
                    else:
                        print(f"   ⚠️ Warning: All {total_pages} pages were skipped. Nothing to process.")
                        continue
            except Exception as e:
                print(f"   ⚠️ Could not read total pages: {e}. Skipping PageRange optimization.")
        
        print(f"⏳ Converting {pdf_path}...")
        try:
            if page_range is not None:
                result = converter.convert(pdf_path, page_range=page_range)
            else:
                result = converter.convert(pdf_path)
            doc = result.document
        except Exception as e:
            print(f"❌ Failed to convert {pdf_path}: {e}")
            continue
       
        print("⚙️ Running Hierarchical Chunker...")
       
        # Map item ID to document index for sorting
        item_id_to_index = {id(item): i for i, (item, _) in enumerate(doc.iterate_items())}
       
        chunker = HierarchicalChunker()
        final_output = []
        processed_item_ids = set()
       
        # --- Pass 1: Standard Hierarchical Chunks ---
        for chunk in chunker.chunk(doc):
            context = get_context_string(chunk.meta.headings)
            items = chunk.meta.doc_items
           
            # Filter logic: Check if ANY valid item in the chunk belongs to a skipped page
            # If the chunk is purely from a skipped page, we skip it.
            # If it spans (unlikely for standard text), we might keep it but usually chunks are localized.
           
            # Let's check the provenance of the FIRST item to decide (chunks usually don't span pages widely)
            # Better: check if ALL items are on skipped pages -> reject
            # Even better: The user wants to remove content from those pages.
           
            # We will process items. If an item is on a skipped page, we ignore it.
            # If a chunk becomes empty because all its items were skipped, we don't add it.
           
            valid_items_for_chunk = []
            for item in items:
                # Check item page number (1-based index in Docling)
                # item.prov is a list of provenance items usually.
                # We check the first provenance for page no.
                if hasattr(item, 'prov') and item.prov:
                    page_no = item.prov[0].page_no
                    if page_no in skip_pages:
                        continue # Skip this item
               
                valid_items_for_chunk.append(item)
           
            if not valid_items_for_chunk:
                continue # Chunk is empty after filtering
               
            has_table = any(isinstance(i, TableItem) for i in valid_items_for_chunk)
           
            if has_table:
                for item in valid_items_for_chunk:
                    if isinstance(item, TableItem):
                        sort_idx = item_id_to_index.get(id(item), 999999)
                        table_chunks = process_table_item(item, doc, context)
                       
                        for tc in table_chunks:
                            tc["_sort_index"] = sort_idx
                            final_output.append(tc)
                       
                        processed_item_ids.add(id(item))
                    else:
                        processed_item_ids.add(id(item))
            else:
                # Logic: If the chunk belongs to a skipped page, drop it.
               
                chunk_page = -1
                chunk_prov = getattr(chunk, 'prov', None)
                if chunk_prov and chunk_prov[0]:
                    chunk_page = chunk_prov[0].page_no
               
                if chunk_page in skip_pages:
                    # The chunk starts on a skipped page.
                    # Most chunks are single-page or mostly single-page.
                    continue
               
                text = chunk.text.strip()
                if text:
                    # Use index of first item for sorting
                    first_id = id(valid_items_for_chunk[0]) if valid_items_for_chunk else 0
                    sort_idx = item_id_to_index.get(first_id, 999999)
                   
                    final_output.append({
                        "text": text,
                        "context": context,
                        "_sort_index": sort_idx
                    })
               
                # Mark all original items as processed so Pass 2 doesn't pick them up
                for item in items:
                    processed_item_ids.add(id(item))
 
        # --- Pass 2: Catch-up for Missed Headers & Content ---
        print("   - Catching up on missed headers/content...")
       
        # We need to track context manually for these missed items
        current_header_stack = []
        added_count = 0
       
        for item, level in doc.iterate_items():
           
            # Determine page number
            item_page = -1
            if hasattr(item, 'prov') and item.prov:
                item_page = item.prov[0].page_no
           
            # Maintain header stack regardless of skipping (to keep context correct for subsequent pages)
            if isinstance(item, SectionHeaderItem):
                header_text = item.text.strip()
                while len(current_header_stack) >= level:
                    current_header_stack.pop()
                current_header_stack.append(header_text)
           
            # SKIP LOGIC: If item is on skipped page, do not add it.
            if item_page in skip_pages:
                continue
 
            # Check if missed
            if id(item) not in processed_item_ids:
                # We specifically want to include Headers now, plus any other missed text
                if isinstance(item, (SectionHeaderItem, TextItem, ListItem)):
                        text = item.text.strip()
                        if text:
                            eff_context = ""
                           
                            # Determine effective stack for this item
                            if isinstance(item, SectionHeaderItem):
                                # For a header, its context is its parents represented by the stack EXCLUDING itself (last item)
                                relevant_stack = current_header_stack[:-1]
                            else:
                                # For text, context is the full current stack
                                relevant_stack = current_header_stack
                           
                            # Build context string from the last 2 items of the relevant stack
                            if relevant_stack:
                                eff_context = " > ".join(relevant_stack[-2:])
 
                            sort_idx = item_id_to_index.get(id(item), 999999)
                           
                            final_output.append({
                            "text": text,
                            "context": eff_context,
                            "_sort_index": sort_idx
                            })
                            added_count += 1
                            processed_item_ids.add(id(item))
 
        print(f"   - Added {added_count} items (Headers/Missed Text) that were skipped by standard chunking.")
 
        # 3. Deduplicate
        print("   - Removing content-based duplicates (Text + Context)...")
        unique_final_output = []
        seen_content = set()
        duplicates_removed = 0
       
        for chunk in final_output:
            # Create a unique signature for the chunk content
            signature = (chunk.get("text", "").strip(), chunk.get("context", "").strip())
           
            if signature not in seen_content:
                unique_final_output.append(chunk)
                seen_content.add(signature)
            else:
                duplicates_removed += 1
               
        print(f"   - Removed {duplicates_removed} duplicate chunks.")
 
        # 4. Sort and Save
        print("   - Dropping chunks with empty context...")
        unique_final_output = [c for c in unique_final_output if c.get("context", "").strip()]
       
        print("   - Sorting chunks by document order...")
        unique_final_output.sort(key=lambda x: x.get("_sort_index", 999999))
       
        # Clean up sort key
        for chunk in unique_final_output:
            chunk.pop("_sort_index", None)
 
        output_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}.json"
        output_path = os.path.join(CHUNKS_DIR, output_filename)
       
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(unique_final_output, f, indent=4, ensure_ascii=False)
            
        print(f"💾 Saved {len(unique_final_output)} chunks to {output_path}")

        # 5. Explicitly clear memory
        del converter
        del pipeline_options
        gc.collect()
 
if __name__ == "__main__":
    main()