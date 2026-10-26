"""Serve the local raw-footage reviewer with byte-range video seeking."""
import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('data/qa/sequence-analytics-2026-10-25/benchmark'))
    parser.add_argument('--port',type=int,default=8877)
    args=parser.parse_args()
    directory=args.directory.resolve()
    if not (directory/'review.html').is_file():raise ValueError('Prepare the benchmark first')
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    app.mount('/',StaticFiles(directory=str(directory),html=True))
    uvicorn.run(app,host='127.0.0.1',port=args.port)
