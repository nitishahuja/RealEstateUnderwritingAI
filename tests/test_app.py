import unittest
from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict(self):
        response = self.app.post("/predict", json={
            "ZIP_Code": 0,
            "Property_Type": 1,
            "Square_Footage": 0.5,
            "Asking_Price": 0.7,
            "Estimated_Rent": 0.6,
            "Vacancy_Rate": 0.05,
            "NOI": 0.4,
            "Cap_Rate": 0.06
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("predicted_cash_flow", response.get_json())

if __name__ == "__main__":
    unittest.main()
