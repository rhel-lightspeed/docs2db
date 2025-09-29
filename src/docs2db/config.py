"""Settings for the project."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Authentication credentials for the GraphQL API
    client_id: str = ""
    client_secret: str = ""

    # Internal CA certificates for GraphQL API
    ca_cert_file: str = "certs/ca.pem"

    # URLs for authentication and GraphQL API
    auth_url: str = "https://sso.redhat.com/auth/realms/redhat-external/protocol/openid-connect/token"
    graphql_url: str = "https://vpn.graphql.redhat.com/"

    # Base directory for storing content
    content_base_dir: str = "content"

    # Directory for storing cached GraphQL queries
    graphql_query_dir: str = "queries"

    # GraphQL query timeout in seconds
    graphql_timeout: int = 60

    # The auth token expires after 900 seconds (15 minutes), so this number should be
    # less than that.s
    auth_token_cache_time: int = 300

    # Set to True if you want to overwrite existing content with new content.
    overwrite_content: bool = False

    # List of documentation slugs to process. The slug corresponds to the documentation URL
    # on redhat.com. For example, the slug "red_hat_enterprise_linux" corresponds to:
    # https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10
    graphql_docs_slugs: list = [
        # OpenShift
        "openshift_container_platform",
        "red_hat_openshift_lightspeed",
        "red_hat_openshift_service_on_aws",
        # Ansible
        "red_hat_ansible_automation_platform",
        # RHEL
        "red_hat_enterprise_linux",
        "red_hat_enterprise_linux_ai",
        "red_hat_enterprise_linux_for_real_time",
        "red_hat_enterprise_linux_for_sap_solutions",
        "red_hat_insights",
        "red_hat_satellite",
    ]


settings = Settings()
