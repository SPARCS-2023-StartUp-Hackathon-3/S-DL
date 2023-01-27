from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List
import boto3
from botocore.client import Config as S3Config
from config import *

app = FastAPI()

s3_config = S3Config(signature_version="s3v4")

s3 = boto3.resource(
    "s3",
    aws_access_key_id=NEXT_PUBLIC_S3_ACCESS_KEY_ID,
    aws_secret_access_key=NEXT_PUBLIC_S3_SECRET_ACCESS_KEY,
    config=s3_config
)


class GenerateItem(BaseModel):
    title: str
    color: str
    desc: str

    def get_prompt(self):
        return f"title: {self.title}\ncolor: {self.color}\nstyle: {self.desc}"


class EditItem(GenerateItem):
    image: str


class ResponseItem(BaseModel):
    images: List[str]


@app.post("/v1/generate", response_model=ResponseItem)
async def generate(item: GenerateItem):
    return {"images": [
        "20230127220712340.jpg",
        "20230127220712341.jpg",
        "20230127220712342.jpg",
        "20230127220712343.jpg",
        "20230127220712344.jpg",
        "20230127220712345.jpg",
    ]}


@app.post("/v1/edit", response_model=ResponseItem)
async def edit(item: EditItem):
    return {"images": [
        "20230127220712340.jpg"
    ]}
