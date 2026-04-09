"""Tests for HTML filing extraction."""
import pytest
from finchartaudit.tools.html_extract import HtmlFilingExtractor


@pytest.fixture
def extractor():
    return HtmlFilingExtractor()


class TestHtmlExtraction:
    def test_extract_tables(self, extractor):
        html = """<html><body>
        <table><tr><th>Metric</th><th>2023</th></tr>
        <tr><td>Revenue</td><td>$1,234M</td></tr>
        <tr><td>EBITDA</td><td>$456M</td></tr></table>
        </body></html>"""
        result = extractor.extract_from_string(html)
        assert len(result["tables"]) >= 1
        assert any("Revenue" in str(t) for t in result["tables"])

    def test_extract_text(self, extractor):
        html = "<html><body><p>Adjusted EBITDA margin improved to 25.3% in 2023.</p></body></html>"
        result = extractor.extract_from_string(html)
        assert "Adjusted EBITDA" in result["text"]

    def test_detect_nongaap_terms(self, extractor):
        html = "<html><body><p>Non-GAAP operating income was $500M. Adjusted EBITDA reached $600M.</p></body></html>"
        result = extractor.extract_from_string(html)
        assert len(result["nongaap_mentions"]) >= 1

    def test_extract_from_file(self, extractor, tmp_path):
        html_file = tmp_path / "test.htm"
        html_file.write_text("<html><body><p>Test content</p></body></html>", encoding="utf-8")
        result = extractor.extract_from_file(str(html_file))
        assert "Test content" in result["text"]
