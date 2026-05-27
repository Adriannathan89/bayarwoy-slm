import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestPreprocess:
    def test_lowercase_and_strip(self):
        from classifier import preprocess
        assert preprocess("  Makan SIANG  ") == "makan siang"

    def test_removes_stopwords(self):
        from classifier import preprocess
        result = preprocess("makan yang enak")
        tokens = result.split()
        assert "yang" not in tokens
        assert "makan" in tokens
        assert "enak" in tokens

    def test_keeps_ke_dan_dari(self):
        from classifier import preprocess
        result = preprocess("transfer ke rekening dari ATM")
        tokens = result.split()
        assert "ke" in tokens
        assert "dari" in tokens

    def test_removes_punctuation(self):
        from classifier import preprocess
        assert preprocess("makan! siang.") == "makan siang"

    def test_removes_short_tokens(self):
        from classifier import preprocess
        result = preprocess("makan a b")
        tokens = result.split()
        assert "a" not in tokens
        assert "b" not in tokens
        assert "makan" in tokens


class TestMoneyDirectionFeatures:
    def setup_method(self):
        from classifier import MoneyDirectionFeatures
        self.mdf = MoneyDirectionFeatures()

    def test_output_shape(self):
        out = self.mdf.transform(["bayar listrik", "dapat gaji"])
        assert out.shape == (2, 8)

    def test_income_cue_n_in(self):
        out = self.mdf.transform(["dapat gaji bulan ini"]).toarray()
        assert out[0, 0] > 0, "kolom n_in harus > 0 untuk cue 'dapat'"

    def test_expense_cue_n_out(self):
        out = self.mdf.transform(["bayar listrik PLN"]).toarray()
        assert out[0, 1] > 0, "kolom n_out harus > 0 untuk cue 'bayar'"

    def test_has_ke(self):
        out = self.mdf.transform(["transfer ke kakak"]).toarray()
        assert out[0, 4] == 1, "has_ke harus 1"

    def test_has_dari(self):
        out = self.mdf.transform(["dikasih uang dari ayah"]).toarray()
        assert out[0, 5] == 1, "has_dari harus 1"

    def test_has_sama(self):
        out = self.mdf.transform(["makan sama teman"]).toarray()
        assert out[0, 6] == 1, "has_sama harus 1"

    def test_ambiguous_text_no_cues(self):
        out = self.mdf.transform(["komisi tomoro"]).toarray()
        assert out[0, 0] == 0 and out[0, 1] == 0, \
            "teks ambigu tanpa kata kerja arah: n_in dan n_out harus 0"

    def test_starts_in_flag(self):
        out = self.mdf.transform(["dapat bonus dari kantor"]).toarray()
        assert out[0, 2] == 1, "starts_in harus 1 jika dimulai dengan cue masuk"

    def test_starts_out_flag(self):
        out = self.mdf.transform(["bayar tagihan internet"]).toarray()
        assert out[0, 3] == 1, "starts_out harus 1 jika dimulai dengan cue keluar"
