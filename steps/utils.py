import os
import numpy as np
import json

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace

def connectWithAzure() -> Workspace:
    """
        Method that will connect to Azure and return a Workspace
    """
    tenant_id = os.environ.get("TENANT_ID")
    client_id = os.environ.get("CLIENT_ID")
    client_secret = os.environ.get("CLIENT_SECRET")

    # Service Principle Authentication to automate the login. Otherwise you'll have to login with your own user account.
    # Get these parameters from the Azure Portal / Azure CLI
    spa = ServicePrincipalAuthentication(tenant_id=tenant_id,  # tenantID
                                             service_principal_id=client_id,  # clientId
                                             service_principal_password=client_secret)  # clientSecret

    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    return Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=spa
    )

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)