import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configura e retorna um logger estruturado.
    
    Args:
        name (str): Nome do módulo chamador (geralmente __name__).
        
    Returns:
        logging.Logger: Instância configurada do logger.
    """
    logger = logging.getLogger(name)
    
    # Evita adicionar múltiplos handlers se o logger já existir
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formato do log (Data, Nível, Módulo, Mensagem)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Saída para o console (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger