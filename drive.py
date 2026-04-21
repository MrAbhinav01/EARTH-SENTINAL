print("🔥 RUNNING NEW DRIVE SCRIPT")

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import zipfile
import subprocess
import time
import sys

print("🐍 Using Python:", sys.executable)

# ✅ FORCE correct Python (IMPORTANT FIX)
VENV_PYTHON = r"D:\EarthSentinel-master\venv\Scripts\python.exe"

# ✅ Google Drive Folder ID
FOLDER_ID = "1bNWbiAE65U-LXelploL20vPWS5M7Ze6w"

# ✅ Local folder
LOCAL_FOLDER = "downloaded_weeks"
os.makedirs(LOCAL_FOLDER, exist_ok=True)

# ---- AUTH ----
print("🔑 Authenticating Google Drive...")
gauth = GoogleAuth()
gauth.settings["client_config_file"] = "client_secrets.json"
gauth.settings["get_refresh_token"] = True

gauth.LoadCredentialsFile("credentials.json")

if gauth.credentials is None:
    print("⚠️ No credentials → login required")
    gauth.CommandLineAuth()
elif gauth.access_token_expired:
    print("♻️ Token expired → login again")
    gauth.CommandLineAuth()
else:
    print("✅ Using cached credentials")

gauth.SaveCredentialsFile("credentials.json")
drive = GoogleDrive(gauth)

# ---- CHECK FILES ----
print("\n📂 Checking Drive folder...")

MAX_RETRIES = 10
WAIT_SECONDS = 10

for attempt in range(MAX_RETRIES):
    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    if len(file_list) > 0:
        print(f"✅ Found {len(file_list)} files")
        break

    print(f"⏳ Retry {attempt+1}/{MAX_RETRIES}")
    time.sleep(WAIT_SECONDS)

if len(file_list) == 0:
    raise ValueError("❌ No files found.")

# ---- LIST FILES ----
print("\n📄 Files:")
for f in file_list:
    print(" -", f['title'])

# ---- DOWNLOAD ----
print("\n⬇ Downloading...")
for f in file_list:
    path = os.path.join(LOCAL_FOLDER, f['title'])

    if os.path.exists(path):
        print("✅ Exists:", f['title'])
        continue

    try:
        print("⬇", f['title'])
        f.GetContentFile(path)
    except Exception as e:
        print("❌ Failed:", f['title'], "|", e)

print("✅ Download done")

# ---- EXTRACT ----
print("\n🗜 Extracting...")
for file in os.listdir(LOCAL_FOLDER):
    if file.endswith(".zip"):
        zip_path = os.path.join(LOCAL_FOLDER, file)
        extract_path = os.path.join(LOCAL_FOLDER, file.replace(".zip", ""))

        if os.path.exists(extract_path):
            print("✅ Already extracted:", file)
            continue

        try:
            print("🗜 Extracting:", file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            print("❌ Extraction failed:", file, "|", e)

print("✅ Extraction done")

VENV_PYTHON = r"D:\EarthSentinel-master\venv\Scripts\python.exe"

subprocess.run([VENV_PYTHON, "create_chunks.py"], check=True)
subprocess.run([VENV_PYTHON, "inference.py"], check=True)
print("\n🎉 PIPELINE COMPLETE")