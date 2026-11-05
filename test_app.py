import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app
import predict_helper


class TestCropYieldApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.app.testing = True
        cls.client = app.app.test_client()

    def test_predict_helper(self):
        """Test standalone predict_helper function across multiple crops."""
        # 2018 Rice
        pred_rice = predict_helper.predict_yield("Andhra Pradesh", "Rice", "Kharif", 2018, 5.0)
        self.assertIsInstance(pred_rice, float)
        self.assertGreater(pred_rice, 0.0)
        print(f"\n[Test] Standalone predict_helper (Rice): {pred_rice:.3f} tonnes/ha")

        # Sugarcane
        pred_sugar = predict_helper.predict_yield("Uttar Pradesh", "Sugarcane", "Whole Year", 2018, 5.0)
        self.assertIsInstance(pred_sugar, float)
        self.assertGreater(pred_sugar, 30.0)
        print(f"[Test] Standalone predict_helper (Sugarcane): {pred_sugar:.3f} tonnes/ha")

        # Cotton
        pred_cotton = predict_helper.predict_yield("Gujarat", "Cotton", "Kharif", 2018, 5.0)
        self.assertIsInstance(pred_cotton, float)
        self.assertGreater(pred_cotton, 0.0)
        self.assertLess(pred_cotton, 5.0)
        print(f"[Test] Standalone predict_helper (Cotton): {pred_cotton:.3f} tonnes/ha")

        # 2025 has no historical weather, checks automated imputation
        pred_imputed = predict_helper.predict_yield("Punjab", "Wheat", "Rabi", 2025, 10.0)
        self.assertIsInstance(pred_imputed, float)
        self.assertGreater(pred_imputed, 0.0)
        print(f"[Test] Standalone predict_helper (2025 imputed weather Wheat): {pred_imputed:.3f} tonnes/ha")

    def test_get_index(self):
        """Test GET route of the Flask application with all 37 crops."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Crop Yield Prediction", html)
        self.assertIn("weather lookup", html.lower())
        self.assertIn("Select state", html)
        self.assertIn("Select district", html)
        self.assertIn("Rice", html)
        self.assertIn("Wheat", html)
        self.assertIn("Cotton", html)
        self.assertIn("Sugarcane", html)
        print("[Test] GET / route rendered successfully with multi-crop options.")

    def test_post_prediction_with_weather(self):
        """Test POST route with historical year (weather matched)."""
        payload = {
            "state": "Andhra Pradesh",
            "district": "Chittoor",
            "crop": "Rice",
            "season": "Kharif",
            "year": "2018",
            "area": "5.0",
        }
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Prediction results", html)
        self.assertIn("Automatic Weather Lookup Applied", html)
        self.assertIn("tonnes/hectare", html)
        self.assertIn("Estimated production", html)
        print("[Test] POST / with weather match succeeded.")

    def test_post_prediction_imputed_weather(self):
        """Test POST route with future year (weather imputed)."""
        payload = {
            "state": "Punjab",
            "district": "Amritsar",
            "crop": "Maize",
            "season": "Kharif",
            "year": "2025",
            "area": "2.5",
        }
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Prediction results", html)
        self.assertIn("Automatic Weather Lookup (Imputed)", html)
        self.assertIn("Estimated production", html)
        print("[Test] POST / with imputed weather succeeded.")

    def test_post_sugarcane_prediction(self):
        """Test POST route predicting high-yield crop (Sugarcane)."""
        payload = {
            "state": "Uttar Pradesh",
            "district": "Meerut",
            "crop": "Sugarcane",
            "season": "Whole Year",
            "year": "2019",
            "area": "10.0",
        }
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Prediction results", html)
        self.assertIn("Estimated production", html)
        print("[Test] POST / with Sugarcane succeeded.")

    def test_post_invalid_input(self):
        """Test POST route with invalid input (negative area)."""
        payload = {
            "state": "Punjab",
            "district": "Amritsar",
            "crop": "Rice",
            "season": "Kharif",
            "year": "2020",
            "area": "-5.0",
        }
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Area must be greater than zero.", html)
        print("[Test] POST / with invalid area triggered validation error correctly.")


if __name__ == "__main__":
    unittest.main()
