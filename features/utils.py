import os


def localDir():
    """获取当前文件所在目录"""
    return os.path.dirname(os.path.abspath(__file__))
