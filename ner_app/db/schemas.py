from datetime import date as date_
from datetime import datetime

from pydantic import BaseModel

"""
定义请求参数模型验证与响应模型验证的Pydantic模型
前端用户键入的句子和模型预测出的entyties都放进数据库
"""

class CreateSentence(BaseModel):
    text: str
    entyties: str


class ReadSentence(CreateSentence):
    id: int
    updated_at: datetime
    created_at: datetime

    class Config:
        orm_mode = True


