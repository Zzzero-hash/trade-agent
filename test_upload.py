#!/usr/bin/env python3
"""
Test script for the file upload endpoint
"""
import asyncio


async def test_file_upload() -> None:
    """Test the file upload endpoint"""

    try:
        # Read the sample CSV file
        with open("sample_data.csv", "rb") as f:
            content = f.read()

        # Create a simple file-like object

        class MockUploadFile:
            def __init__(self, filename, content) -> None:
                self.filename = filename
                self._content = content

            async def read(self):
                return self._content

        file_obj = MockUploadFile("sample_data.csv", content)

        # Test the upload endpoint
        from interface.api.routers.data import upload_csv

        await upload_csv(
            file=file_obj,
            symbol="ETH-USD",
            timeframe="1h"
        )


    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_file_upload())
