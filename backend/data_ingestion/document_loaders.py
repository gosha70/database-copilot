"""
Document loader utilities for ingesting various document types.
"""
import os
import logging
from typing import List, Dict, Optional, Union
import glob

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader
)
import yaml

logger = logging.getLogger(__name__)

def load_text_documents(file_path: str) -> List[Document]:
    """
    Load text documents from a file or directory.
    
    Args:
        file_path: Path to a text file or directory containing text files.
    
    Returns:
        A list of Document objects.
    """
    if os.path.isdir(file_path):
        logger.info(f"Loading text documents from directory: {file_path}")
        loader = DirectoryLoader(
            file_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        return loader.load()
    elif os.path.isfile(file_path) and file_path.endswith(".txt"):
        logger.info(f"Loading text document: {file_path}")
        loader = TextLoader(file_path)
        return loader.load()
    else:
        logger.warning(f"Invalid text file path: {file_path}")
        return []

def load_pdf_documents(file_path: str) -> List[Document]:
    """
    Load PDF documents from a file or directory.
    
    Args:
        file_path: Path to a PDF file or directory containing PDF files.
    
    Returns:
        A list of Document objects.
    """
    if os.path.isdir(file_path):
        logger.info(f"Loading PDF documents from directory: {file_path}")
        loader = DirectoryLoader(
            file_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()
    elif os.path.isfile(file_path) and file_path.endswith(".pdf"):
        logger.info(f"Loading PDF document: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()
    else:
        logger.warning(f"Invalid PDF file path: {file_path}")
        return []

def load_html_documents(file_path: str) -> List[Document]:
    """
    Load HTML documents from a file or directory.
    
    Args:
        file_path: Path to an HTML file or directory containing HTML files.
    
    Returns:
        A list of Document objects.
    """
    if os.path.isdir(file_path):
        logger.info(f"Loading HTML documents from directory: {file_path}")
        loader = DirectoryLoader(
            file_path,
            glob="**/*.html",
            loader_cls=BSHTMLLoader
        )
        return loader.load()
    elif os.path.isfile(file_path) and (file_path.endswith(".html") or file_path.endswith(".htm")):
        logger.info(f"Loading HTML document: {file_path}")
        loader = BSHTMLLoader(file_path)
        return loader.load()
    else:
        logger.warning(f"Invalid HTML file path: {file_path}")
        return []

def load_markdown_documents(file_path: str) -> List[Document]:
    """
    Load Markdown documents from a file or directory.
    
    Args:
        file_path: Path to a Markdown file or directory containing Markdown files.
    
    Returns:
        A list of Document objects.
    """
    if os.path.isdir(file_path):
        logger.info(f"Loading Markdown documents from directory: {file_path}")
        loader = DirectoryLoader(
            file_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        return loader.load()
    elif os.path.isfile(file_path) and file_path.endswith(".md"):
        logger.info(f"Loading Markdown document: {file_path}")
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()
    else:
        logger.warning(f"Invalid Markdown file path: {file_path}")
        return []

import xml.etree.ElementTree as ET

def load_xml_documents(file_path: str) -> List[Document]:
    """
    Load XML documents from a file or directory, chunking Liquibase changelogs by changeSet.
    
    Args:
        file_path: Path to an XML file or directory containing XML files.
    
    Returns:
        A list of Document objects.
    """
    def _load_xml_file(xml_file_path: str) -> List[Document]:
        """
        Load a single XML file and convert it to Document objects, chunking by changeSet if present.
        """
        try:
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            docs = []
            try:
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                # Check for Liquibase changelog
                if root.tag.endswith("databaseChangeLog"):
                    for changeset in root.findall(".//{*}changeSet"):
                        cs_str = ET.tostring(changeset, encoding="unicode")
                        cs_id = changeset.attrib.get("id", "unknown")
                        cs_author = changeset.attrib.get("author", "unknown")
                        metadata = {
                            "source": xml_file_path,
                            "changeSet_id": cs_id,
                            "changeSet_author": cs_author
                        }
                        docs.append(Document(page_content=cs_str, metadata=metadata))
                    if docs:
                        return docs
            except Exception as e:
                logger.warning(f"Error parsing XML for chunking: {xml_file_path}: {e}")
            # Fallback: treat the whole file as one document
            metadata = {"source": xml_file_path}
            docs.append(Document(page_content=content, metadata=metadata))
            return docs
        except Exception as e:
            logger.error(f"Error loading XML file {xml_file_path}: {e}")
            return []

    documents = []
    if os.path.isdir(file_path):
        logger.info(f"Loading XML documents from directory: {file_path}")
        xml_files = glob.glob(os.path.join(file_path, "**/*.xml"), recursive=True)
        for xml_file in xml_files:
            documents.extend(_load_xml_file(xml_file))
        return documents
    elif os.path.isfile(file_path) and file_path.endswith(".xml"):
        logger.info(f"Loading XML document: {file_path}")
        return _load_xml_file(file_path)
    else:
        logger.warning(f"Invalid XML file path: {file_path}")
        return []

def load_yaml_documents(file_path: str) -> List[Document]:
    """
    Load YAML documents from a file or directory, chunking by changeSet if possible.
    
    Args:
        file_path: Path to a YAML file or directory containing YAML files.
    
    Returns:
        A list of Document objects.
    """
    def _load_yaml_file(yaml_file_path: str) -> List[Document]:
        """
        Load a single YAML file and convert it to Document objects, chunking by changeSet if present.
        
        Args:
            yaml_file_path: Path to a YAML file.
            
        Returns:
            A list of Document objects.
        """
        try:
            with open(yaml_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            yaml_data = yaml.safe_load(content)
            docs = []
            # If this is a Liquibase changelog, chunk by changeSet
            if isinstance(yaml_data, dict) and "databaseChangeLog" in yaml_data:
                dbcl = yaml_data["databaseChangeLog"]
                if isinstance(dbcl, list):
                    for entry in dbcl:
                        if isinstance(entry, dict) and "changeSet" in entry:
                            changeSets = entry["changeSet"]
                            if isinstance(changeSets, list):
                                for cs in changeSets:
                                    cs_str = yaml.dump(cs, default_flow_style=False)
                                    cs_id = cs.get("id", "unknown")
                                    cs_author = cs.get("author", "unknown")
                                    metadata = {
                                        "source": yaml_file_path,
                                        "changeSet_id": cs_id,
                                        "changeSet_author": cs_author
                                    }
                                    docs.append(Document(page_content=cs_str, metadata=metadata))
                            elif isinstance(changeSets, dict):
                                cs = changeSets
                                cs_str = yaml.dump(cs, default_flow_style=False)
                                cs_id = cs.get("id", "unknown")
                                cs_author = cs.get("author", "unknown")
                                metadata = {
                                    "source": yaml_file_path,
                                    "changeSet_id": cs_id,
                                    "changeSet_author": cs_author
                                }
                                docs.append(Document(page_content=cs_str, metadata=metadata))
                elif isinstance(dbcl, dict) and "changeSet" in dbcl:
                    changeSets = dbcl["changeSet"]
                    if isinstance(changeSets, list):
                        for cs in changeSets:
                            cs_str = yaml.dump(cs, default_flow_style=False)
                            cs_id = cs.get("id", "unknown")
                            cs_author = cs.get("author", "unknown")
                            metadata = {
                                "source": yaml_file_path,
                                "changeSet_id": cs_id,
                                "changeSet_author": cs_author
                            }
                            docs.append(Document(page_content=cs_str, metadata=metadata))
                    elif isinstance(changeSets, dict):
                        cs = changeSets
                        cs_str = yaml.dump(cs, default_flow_style=False)
                        cs_id = cs.get("id", "unknown")
                        cs_author = cs.get("author", "unknown")
                        metadata = {
                            "source": yaml_file_path,
                            "changeSet_id": cs_id,
                            "changeSet_author": cs_author
                        }
                        docs.append(Document(page_content=cs_str, metadata=metadata))
                else:
                    # Fallback: treat the whole file as one document
                    yaml_str = yaml.dump(yaml_data, default_flow_style=False)
                    metadata = {"source": yaml_file_path}
                    docs.append(Document(page_content=yaml_str, metadata=metadata))
            else:
                # Not a Liquibase changelog, treat the whole file as one document
                yaml_str = yaml.dump(yaml_data, default_flow_style=False)
                metadata = {"source": yaml_file_path}
                docs.append(Document(page_content=yaml_str, metadata=metadata))
            return docs
        except Exception as e:
            logger.error(f"Error loading YAML file {yaml_file_path}: {e}")
            return []
    
    if os.path.isdir(file_path):
        logger.info(f"Loading YAML documents from directory: {file_path}")
        yaml_files = []
        for ext in [".yaml", ".yml"]:
            yaml_files.extend(glob.glob(os.path.join(file_path, f"**/*{ext}"), recursive=True))
        
        documents = []
        for yaml_file in yaml_files:
            documents.extend(_load_yaml_file(yaml_file))
        
        return documents
    elif os.path.isfile(file_path) and (file_path.endswith(".yaml") or file_path.endswith(".yml")):
        logger.info(f"Loading YAML document: {file_path}")
        return _load_yaml_file(file_path)
    else:
        logger.warning(f"Invalid YAML file path: {file_path}")
        return []

def load_java_documents(file_path: str) -> List[Document]:
    """
    Load Java documents from a file or directory.
    
    Args:
        file_path: Path to a Java file or directory containing Java files.
    
    Returns:
        A list of Document objects.
    """
    if os.path.isdir(file_path):
        logger.info(f"Loading Java documents from directory: {file_path}")
        loader = DirectoryLoader(
            file_path,
            glob="**/*.java",
            loader_cls=TextLoader  # Using TextLoader for Java files
        )
        return loader.load()
    elif os.path.isfile(file_path) and file_path.endswith(".java"):
        logger.info(f"Loading Java document: {file_path}")
        loader = TextLoader(file_path)
        return loader.load()
    else:
        logger.warning(f"Invalid Java file path: {file_path}")
        return []

def load_documents(file_path: str) -> List[Document]:
    """
    Load documents from a file or directory based on file extension.
    
    Args:
        file_path: Path to a file or directory.
    
    Returns:
        A list of Document objects.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Path does not exist: {file_path}")
        return []
    
    if os.path.isdir(file_path):
        logger.info(f"Loading documents from directory: {file_path}")
        documents = []
        documents.extend(load_text_documents(file_path))
        documents.extend(load_pdf_documents(file_path))
        documents.extend(load_html_documents(file_path))
        documents.extend(load_markdown_documents(file_path))
        documents.extend(load_xml_documents(file_path))
        documents.extend(load_yaml_documents(file_path))
        documents.extend(load_java_documents(file_path))  # Add Java support
        return documents
    elif os.path.isfile(file_path):
        logger.info(f"Loading document: {file_path}")
        if file_path.endswith(".txt"):
            return load_text_documents(file_path)
        elif file_path.endswith(".pdf"):
            return load_pdf_documents(file_path)
        elif file_path.endswith(".html") or file_path.endswith(".htm"):
            return load_html_documents(file_path)
        elif file_path.endswith(".md"):
            return load_markdown_documents(file_path)
        elif file_path.endswith(".xml"):
            return load_xml_documents(file_path)
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            return load_yaml_documents(file_path)
        elif file_path.endswith(".java"):
            return load_java_documents(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
    else:
        logger.warning(f"Invalid file path: {file_path}")
        return []
