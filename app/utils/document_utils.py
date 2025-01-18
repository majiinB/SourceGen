import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ipaddress import ip_address, AddressValueError

def text_formater(text: str) -> str:
    """
    This function cleans the text by removing newlines, tabs, and extra spaces.
    :param text: The text to be cleaned
    :return cleaned_text: The cleaned text
    """

    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = cleaned_text.replace("\t", " ").strip()
    cleaned_text = cleaned_text.replace("\r", " ").strip()
    cleaned_text = cleaned_text.replace("\n", " ").strip()
    cleaned_text = cleaned_text.replace("  ", " ").strip()
    cleaned_text = cleaned_text.replace("   ", " ").strip()

    return cleaned_text

def multiple_text_formater(text_list: list[str]) -> list[str]:
    """
    This function cleans the list of text by removing newlines, tabs, and extra spaces.
    :param text_list: The list of text to be cleaned
    :return cleaned_text_list: The list of cleaned text
    """
    cleaned_text_list = []
    for text in text_list:
        cleaned_text_list.append(text_formater(text))
    return cleaned_text_list

def get_document_stats(stats : dict) -> pd.DataFrame:
    """
    This function returns the statistics of a page in a dataframe format.
    :param stats: The page to be analyzed
    :return DataFrame: The statistics of the page
    """
    data = pd.DataFrame(stats)
    return data.describe().round(2)

def split_text_with_separators(text: str) -> list[str]:
    """
    This function splits a text using a separator. and a pre-determined chunk size [1000]
    :param text: The text to be split
    :return split_text: The split text
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    split_text=text_splitter.split_text(text)
    return split_text

def is_valid_collection_name(name: str) -> bool:
    """
    Validates a collection name based on the following criteria:
    - Length must be between 3 and 63 characters.
    - Must start and end with a lowercase letter or a digit.
    - Can contain dots, dashes, and underscores in between.
    - Must not contain two consecutive dots.
    - Must not be a valid IP address.

    :param name: The collection name to validate.
    :return: True if valid, False otherwise.
    """
    # Check length
    if not (3 <= len(name) <= 63):
        return False

    # Check start and end conditions
    if not re.match(r"^[a-z0-9].*[a-z0-9]$", name):
        return False

    # Check allowed characters and no consecutive dots
    if not re.match(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$", name) or ".." in name:
        return False

    # Check if the name is a valid IP address
    try:
        ip_address(name)
        return False  # If it's a valid IP address, the name is invalid
    except ValueError:
        pass  # Not an IP address, continue

    return True






