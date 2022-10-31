


from sqlalchemy import Column, String, Integer, BigInteger, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from .database import Base

"""
有哪些表
通过数据库配置文件中的基类来创建模型类
"""
class SentenceEntities(Base):
    __tablename__ = 'sentence_entyties'  # 数据表的表名

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(String(500), unique=True, nullable=False, comment='用户传入语句')
    entyties=Column(String(500), unique=False, nullable=False, comment='命名实体')

    created_at = Column(DateTime, server_default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment='更新时间')
