from glocaltokens.client import GLocalAuthenticationTokens

if __name__ == "__main__":
    client = GLocalAuthenticationTokens(
        master_token="<The master token you created>"
        username="<YOUR_GOOGLE_USERNAME>", # your gmail address
        password="FAKE_PASSWORD" # won't be used
    )

    homegraph_response = client.get_homegraph()
    # This one will list all your home devices
    # One of them would be your Nest Camera
    print(homegraph_response.home.devices)
