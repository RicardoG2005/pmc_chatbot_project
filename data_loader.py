import os
import glob
import xml.etree.ElementTree as ET
from typing import List
from langchain.schema import Document
from tqdm.auto import tqdm

def extract_body(nxml_path: str) -> str:
    '''
    Extracts the body content from an NXML file.
    nxml_path: Path to the NXML file
    Returns: a string concatenated paragraph texts
    '''
    try:
        tree = ET.parse(nxml_path)
        root = tree.getroot()
        paras = [
            "".join(p.itertext())
            for p in root.findall(".//{*}body//{*}p")
        ]
        return "\n\n".join(paras)
    except ET.ParseError as e:
        print(f"[ParseError] Skipping file {nxml_path}: {e}")
        return ""
    except Exception as e:
        print(f"[Error] Skipping file {nxml_path}: {e}")
        return ""
    
def load_documents(folder_path: str, limit: int = 500) -> List[Document]:
    """
    Loads and parses NXML files from a folder into LangChain Document objects.
    Args: 
    folder_path: Path to the folder containing NXML files
    limit: Max number of files to process (default 500)

    Returns:
    A list of LangChain Document objects with metadata
    """
    paths = glob.glob(os.path.join(folder_path, "*.xml"))
    print(f"[Loader] Found {len(paths)} XML files.")

    subset_paths = paths[:limit] # only get the limit of files
    print(f"[Loader] Parsing {len(subset_paths)} files...")

    documents = []
    for path in tqdm(subset_paths, desc = "Parsing XML"):
        content = extract_body(path)
        if content.strip():
            documents.append(Document(page_content = content, metadata = {"source": path}))
        
    print(f"[Loader] Finished loading {len(documents)} documents.")
    return documents