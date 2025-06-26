from src.utils.constant import RFMID_PATH

def get_data_path(data_name:str) -> list:
    """Get data storage path for dataset query

    Args:
        data_name (str): Query

    Returns:
        list: data directory list
    """
    if isinstance(data_name, str):
        data_name = [data_name]
    for name in data_name:
        name = name.lower()
        if name =="rfmid":
            return RFMID_PATH
        else:
            raise ValueError(f"Error. Query dataset {name} is currently not supported.")
