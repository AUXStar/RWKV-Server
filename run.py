from server.api.app import create_app
from uvicorn import run


app = create_app()

run(app,host="0.0.0.0")