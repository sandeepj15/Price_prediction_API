# main.py
import datetime as dt
from typing import Dict, Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2, OAuth2PasswordRequestForm
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.handlers.sha2_crypt import sha512_crypt as crypto
from pydantic import BaseModel
from rich.console import Console
from fastapi.staticfiles import StaticFiles
from bitcoin_prediction import integrate_prediction
import pandas as pd
from starlette.responses import RedirectResponse
from starlette import status
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

console = Console()

logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO  # Set the desired logging level
)

# --------------------------------------------------------------------------
# Models and Data
# --------------------------------------------------------------------------
class User(BaseModel):
    username: str
    hashed_password: str


class Settings:
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
    COOKIE_NAME: str = os.getenv("COOKIE_NAME")


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static"), name="static")
settings = Settings()

# Connect to MongoDB using environment variables
mongo_uri = os.getenv("MONGODB_URI")
mongo_db_name = os.getenv("MONGODB_DB")

mongo_client = AsyncIOMotorClient(mongo_uri)
mongo_db = mongo_client[mongo_db_name]
users_collection = mongo_db["users"]

async def init_db():
    # Check if collections exist, create them if not
    if "users" not in (await mongo_db.list_collection_names()):
        await mongo_db.create_collection("users")
        # Insert initial user data if needed


# --------------------------------------------------------------------------
# Sample Logging Usage
# --------------------------------------------------------------------------
def sample_logging_usage():
    try:
        # Some code that might raise an exception
        raise ValueError("This is a sample error.")
    except Exception as e:
        # Log the exception
        logging.error(f"An error occurred: {str(e)}", exc_info=True)



# --------------------------------------------------------------------------
# Registration Page
# --------------------------------------------------------------------------

class RegisterForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.errors: list = []
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.confirm_password: Optional[str] = None

    async def load_data(self):
        form = await self.request.form()
        self.username = form.get("username")
        self.password = form.get("password")
        self.confirm_password = form.get("confirm_password")

    async def is_valid(self):
        if not self.username or not (self.username.__contains__("@")):
            self.errors.append("Email is required")
        if not self.password or not len(self.password) >= 4:
            self.errors.append("A valid password is required")
        if self.password != self.confirm_password:
            self.errors.append("Passwords do not match")
        if not self.errors:
            return True
        return False

@app.post("/auth/register", response_class=HTMLResponse)
async def register_post(request: Request):
    form = RegisterForm(request)
    await form.load_data()
    if await form.is_valid():
        # Hash the password before storing it
        hashed_password = crypto.hash(form.password)

        # Check if the user already exists
        existing_user = await users_collection.find_one({"username": form.username})
        if existing_user:
            form.errors.append("User already exists")
            return templates.TemplateResponse("register.html", form.__dict__)

        # Insert the new user into the database
        await users_collection.insert_one({"username": form.username, "hashed_password": hashed_password})

        # Redirect to the login page after successful registration
        response = RedirectResponse("/auth/login", status.HTTP_302_FOUND)
        return response

    return templates.TemplateResponse("register.html", form.__dict__)

# --------------------------------------------------------------------------
# Authentication logic
# --------------------------------------------------------------------------
class OAuth2PasswordBearerWithCookie(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(
            flows=flows,
            scheme_name=scheme_name,
            description=description,
            auto_error=auto_error,
        )

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.cookies.get(settings.COOKIE_NAME)
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_302_FOUND,
                    detail="Not authenticated",
                    headers={"Location": "/auth/login"},
                )
            else:
                return None
        return param


oauth2_scheme = OAuth2PasswordBearerWithCookie(tokenUrl="token")


def create_access_token(data: Dict) -> str:
    to_encode = data.copy()
    expire = dt.datetime.utcnow() + dt.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


async def authenticate_user(username: str, plain_password: str) -> Optional[User]:
    user = await users_collection.find_one({"username": username})
    if not user or not crypto.verify(plain_password, user["hashed_password"]):
        return None
    return User(**user)


def decode_token(token: Optional[str]) -> Optional[User]:
    if token is None:
        return None

    try:
        token = token.removeprefix("Bearer").strip()
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str =  payload.get("username")
        if username is None:
            return None
    except JWTError as e:
        print(e)
        return None

    user =  users_collection.find_one({"username": username})
    return user




async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    user_dict = await decode_token(token)
    if user_dict is None:
        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            detail="Could not validate credentials.",
            headers={"Location": "/auth/login"},
        )
    return User(**user_dict)



def get_current_user_from_cookie(request: Request) -> User:
    token = request.cookies.get(settings.COOKIE_NAME)
    user = decode_token(token)
    return user


@app.post("/token")
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, str]:
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"username": user.username})

    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=f"Bearer {access_token}",
        httponly=True
    )
    return {settings.COOKIE_NAME: access_token, "token_type": "bearer"}


# --------------------------------------------------------------------------
# Home Page
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    try:
        user = get_current_user_from_cookie(request)
    except:
        user = None
    context = {
        "user": user,
        "request": request,
    }
    return templates.TemplateResponse("index.html", context)


# --------------------------------------------------------------------------
# Private Page
# --------------------------------------------------------------------------
@app.get("/private", response_class=HTMLResponse)
def private(request: Request, user: User = Depends(get_current_user_from_token)):
    context = {
        "user": user,
        "request": request
    }
    return templates.TemplateResponse("private.html", context)


# --------------------------------------------------------------------------
# Login - GET
# --------------------------------------------------------------------------
@app.get("/auth/login", response_class=HTMLResponse)
def login_get(request: Request, user: User = Depends(get_current_user_from_cookie)):
    if user:
        # Redirect to the home page if the user is already authenticated
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

    context = {
        "request": request,
    }
    return templates.TemplateResponse("login.html", context)

#-----------------------------------------------------------------
# Register page
#----------------------------------------------------------------
@app.get("/auth/register", response_class=HTMLResponse)
def register_get(request: Request, user: User = Depends(get_current_user_from_cookie)):
    if user:
        # Redirect to the home page if the user is already authenticated
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

    context = {
        "request": request,
    }
    return templates.TemplateResponse("register.html", context)

# --------------------------------------------------------------------------
# Login - POST
# --------------------------------------------------------------------------
class LoginForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.errors: list = []
        self.username: Optional[str] = None
        self.password: Optional[str] = None

    async def load_data(self):
        form = await self.request.form()
        self.username = form.get("username")
        self.password = form.get("password")

    async def is_valid(self):
        if not self.username or not (self.username.__contains__("@")):
            self.errors.append("Email is required")
        if not self.password or not len(self.password) >= 4:
            self.errors.append("A valid password is required")
        if not self.errors:
            return True
        return False


@app.post("/auth/login", response_class=HTMLResponse)
async def login_post(request: Request):
    form = LoginForm(request)
    await form.load_data()
    if await form.is_valid():
        try:
            response = RedirectResponse("/", status.HTTP_302_FOUND)
            await login_for_access_token(response=response, form_data=form)  # Await the function
            form.__dict__.update(msg="Login Successful!")
            console.log("[green]Login successful!!!!")
            return response
        except HTTPException:
            form.__dict__.update(msg="")
            form.__dict__.get("errors").append("Incorrect Email or Password")
            return templates.TemplateResponse("login.html", form.__dict__)
    return templates.TemplateResponse("login.html", form.__dict__)


# --------------------------------------------------------------------------
# Logout
# --------------------------------------------------------------------------
@app.get("/auth/logout", response_class=HTMLResponse)
def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie(settings.COOKIE_NAME)
    return response

#-----------------------------------------------------------------------------
#Price Prediction
#-----------------------------------------------------------------------------
@app.get("/prediction-chart", response_class=HTMLResponse)
async def render_prediction_chart(request: Request, user: User = Depends(get_current_user_from_token)):
    # Use the existing functions to get data and train the model
    fig_json = integrate_prediction()

    # Pass the JSON data and user to the HTML template
    return templates.TemplateResponse("prediction_chart.html", {"request": request, "fig_json": fig_json, "user": user})

# --------------------------------------------------------------------------
# Sample Logging Usage
# --------------------------------------------------------------------------
sample_logging_usage()
app.add_event_handler("startup", init_db)