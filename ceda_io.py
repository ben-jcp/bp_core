"""
===================

Python script for reading a NetCDF file remotely from the CEDA archive. It demonstrates fetching
and using a download token to authenticate access to CEDA Archive data, as well as how to load
and subset the Dataset from a stream of data (diskless), without having to download the whole file.

You will be prompted to provide your CEDA username and password the first time the script is run and
again if the token cached from a previous attempt has expired.
"""
"""
BP Comments May 2025:
Source: https://github.com/cedadev/opendap-python-example/tree/master (extended to accommodate xarray)
"""

import json
import os
import requests
import xarray as xr

from base64 import b64encode
from datetime import datetime, timezone
from getpass import getpass
from netCDF4 import Dataset
from urllib.parse import urlparse


# URL for the CEDA Token API service
TOKEN_URL = "https://services.ceda.ac.uk/api/token/create/"
# Location on the filesystem to store a cached download token
TOKEN_CACHE = os.path.expanduser(os.path.join("~", ".cedatoken"))


def load_cached_token():
    """Read the token back out from its cache file.

    Returns a tuple containing the token and its expiry timestamp
    """

    # Read the token back out from its cache file
    try:
        with open(TOKEN_CACHE, "r") as cache_file:
            data = json.loads(cache_file.read())

            token = data.get("access_token")
            expires = datetime.strptime(data.get("expires"), "%Y-%m-%dT%H:%M:%S.%f%z")
            return token, expires

    except FileNotFoundError:
        return None, None


def get_token():
    """Fetches a download token, either from a cache file or
     from the token API using CEDA login credentials.

    Returns an active download token
    """

    # Check the cache file to see if we already have an active token
    token, expires = load_cached_token()

    # If no token has been cached or the token has expired, we get a new one
    now = datetime.now(timezone.utc)
    if not token or expires < now:

        if not token:
            print(f"No previous token found at {TOKEN_CACHE}. ", end="")
        else:
            print(f"Token at {TOKEN_CACHE} has expired. ", end="")
        print("Generating a fresh token...")

        print("Please provide your CEDA username: ", end="")
        username = input()
        password = getpass(prompt="CEDA user password: ")

        credentials = b64encode(f"{username}:{password}".encode("utf-8")).decode(
            "ascii"
        )
        headers = {
            "Authorization": f"Basic {credentials}",
        }
        response = requests.request("POST", TOKEN_URL, headers=headers)
        if response.status_code == 200:

            # The token endpoint returns JSON
            response_data = json.loads(response.text)
            token = response_data["access_token"]

            # Store the JSON data in the cache file for future use
            with open(TOKEN_CACHE, "w") as cache_file:
                cache_file.write(response.text)

        else:
            print("Failed to generate token, check your username and password.")

    else:
        print(f"Found existing token at {TOKEN_CACHE}, skipping authentication.")

    return token, expires


def open_dataset(url, download_token=None):
    """Open a NetCDF dataset from a remote file URL. Files requiring authentication
     will require an active download token associated with an authorised CEDa user.

    Returns a Python NetCDF4 Dataset object
    """

    headers = None
    # Add the download token to the request header if one is available
    if download_token:
        headers = {"Authorization": f"Bearer {download_token}"}

    response = requests.request("GET", url, headers=headers, stream=True)
    if response.status_code != 200:
        print(
            f"Failed to fetch data. The response from the server was {response.status_code}"
        )
        return
    # print (response.content)

    filename = os.path.basename(urlparse(url).path)
    print(f"Opening Dataset from file {filename} ...")
    # To avoid downloading the whole file, we create an "in-memory" Dataset from the response
    # See: https://unidata.github.io/netcdf4-python/#in-memory-diskless-datasets
    # try:
    #     output_ds = Dataset(filename, memory=response.content)
    # except OSError:
    #     print('Failed to load dataset on existing token, removing and generating new token')
    #     os.remove('~/.cedatoken')
        
    return Dataset(filename, memory=response.content)


def open_ceda_dataset(url, use_xarray=True, token=None):
    """
    BP 29/05/25
    Function will prompt for CEDA Archive login details on first run in a while
    """
    if not token:
        token, _ = get_token()

    dataset = open_dataset(url, download_token=token)

    if use_xarray:
        dataset = xr.load_dataset(xr.backends.NetCDF4DataStore(dataset))

    return dataset