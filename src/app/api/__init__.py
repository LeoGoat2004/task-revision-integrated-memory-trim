"""API layer: FastAPI surface.

Routes are small dispatchers that delegate to the services layer. All
request/response shapes are defined here as Pydantic models; the contract is
frozen at the wire boundary.
"""
