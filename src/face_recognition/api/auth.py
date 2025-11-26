"""Authentication service for SmartOffice API."""

from typing import Optional
import requests
import ssl
import urllib3
from loguru import logger

# Disable SSL verification globally for development
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class AuthenticationService:
    """Handles authentication with the SmartOffice API.

    This service manages login, token storage, and session management.
    """

    def __init__(self, base_url: str, email: str, password: str):
        """Initialize the authentication service.

        Args:
            base_url: Base URL of the API
            email: User email for authentication
            password: User password for authentication
        """
        self.base_url = base_url
        self.email = email
        self.password = password
        self.session = requests.Session()
        # Disable SSL verification for development (if HTTPS is used)
        self.session.verify = False
        # Suppress InsecureRequestWarning
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.token: Optional[str] = None

    def login(self) -> Optional[str]:
        """Authenticate user and obtain access token.

        Returns:
            Access token if successful, None otherwise

        Raises:
            requests.exceptions.RequestException: If login fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"email": self.email, "password": self.password}
            )

            if response.status_code != 200:
                logger.critical(
                    f"Login request failed with status {response.status_code}: {response.text}"
                )
                return None

            data = response.json()

            if data.get('success'):
                self.token = data.get('token')
                # Set authorization header for future requests
                self.session.headers.update({
                    'Authorization': f'Bearer {self.token}'
                })
                logger.info("Successfully authenticated with SmartOffice API")
                return self.token
            else:
                raise requests.exceptions.RequestException(
                    f"Login failed: {data.get('error', 'Unknown error')}"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Login request failed: {str(e)}")
            raise requests.exceptions.RequestException(f"Login request failed: {str(e)}")

    def is_authenticated(self) -> bool:
        """Check if the service is authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        return self.token is not None

    def get_session(self) -> requests.Session:
        """Get the authenticated session.

        Returns:
            Authenticated requests session
        """
        return self.session

    def get_token(self) -> Optional[str]:
        """Get the current access token.

        Returns:
            Access token if authenticated, None otherwise
        """
        return self.token

    def refresh_token(self) -> bool:
        """Refresh the authentication token by logging in again.

        Returns:
            True if token refresh was successful, False otherwise
        """
        logger.info("Refreshing authentication token...")
        try:
            new_token = self.login()
            if new_token:
                logger.info("Token refreshed successfully")
                return True
            else:
                logger.error("Token refresh failed")
                return False
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
