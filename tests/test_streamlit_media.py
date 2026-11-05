"""
The document preview (st.image) must survive Streamlit's own HTTP middleware.

Regression (2026-11-05, production): Starlette 1.8.0 changed GZipMiddleware so
that, for content types it does not compress (images), the response headers
are sent from the "http.response.start" branch. Streamlit 1.58's
SelectiveGZipMiddleware overrides exactly that branch, so the headers were
never sent: uvicorn raised "Expected ASGI message 'http.response.start', but
got 'http.response.body'" and every preview image answered 500.

This test runs a JPEG through Streamlit's real middleware stack, the one
`streamlit run` installs, so a dependency bump that breaks images fails CI
instead of the live site. It does not start Streamlit: only the middleware
and one route.
"""
import pytest

pytest.importorskip("httpx")  # starlette's TestClient drives the app over httpx

from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from streamlit.web.server.starlette.starlette_app import create_streamlit_middleware

JPEG_BODY = b"\xff\xd8\xff\xe0" + b"\x00" * 4096  # big enough to be a gzip candidate


async def _image(request):
    return Response(JPEG_BODY, media_type="image/jpeg")


async def _text(request):
    return Response("x" * 4096, media_type="text/plain")


@pytest.fixture()
def client():
    app = Starlette(
        routes=[Route("/media/preview.jpg", _image), Route("/media/report.txt", _text)],
        middleware=create_streamlit_middleware(),
    )
    # raise_server_exceptions=False: a middleware crash must show up as the
    # 500 a browser gets, not as a test-runner exception.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("accept_encoding", ["gzip, deflate, br, zstd", "identity"])
def test_preview_image_is_served(client, accept_encoding):
    response = client.get("/media/preview.jpg", headers={"Accept-Encoding": accept_encoding})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == JPEG_BODY


def test_compressible_content_still_works(client):
    response = client.get("/media/report.txt", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.text == "x" * 4096
