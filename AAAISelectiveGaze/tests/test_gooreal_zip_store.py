import io
import zipfile

from PIL import Image

from AAAISelectiveGaze.scripts.cache_predictions import _create_gooreal_image_store


def test_nested_gooreal_zip_is_opened_and_cached(tmp_path):
    relative_path = "finalrealdatasetImgsV3/49/cam1/frame.jpg"
    image_bytes = io.BytesIO()
    Image.new("RGB", (7, 5), color=(12, 34, 56)).save(image_bytes, format="JPEG")

    inner_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_bytes, "w") as inner_zip:
        inner_zip.writestr(relative_path, image_bytes.getvalue())
    with zipfile.ZipFile(tmp_path / "gooreal.zip", "w") as outer_zip:
        outer_zip.writestr("finalrealdatasetImgsV3.zip", inner_bytes.getvalue())

    cache_dir = tmp_path / "zip-cache"
    store = _create_gooreal_image_store(tmp_path, zip_cache_dir=cache_dir)
    try:
        image = store.open(relative_path)
        assert image.size == (7, 5)
        assert image.mode == "RGB"
    finally:
        store.close()

    assert (cache_dir / "finalrealdatasetImgsV3.zip").is_file()
