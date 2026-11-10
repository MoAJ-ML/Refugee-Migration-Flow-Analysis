import unittest
import pandas as pd
from validation import rolling_origin_forecasts, mean_absolute_errors


class ValidationTests(unittest.TestCase):
    def test_expanding_window_baseline(self):
        years = pd.date_range('2000-01-01', periods=9, freq='YS')
        series = pd.Series(range(9), index=years, dtype=float)
        result = rolling_origin_forecasts(series, order=(0, 1, 0), first_test_position=5)
        self.assertEqual(result['Year'].tolist(), [2005, 2006, 2007, 2008])
        self.assertEqual(result['Naive'].tolist(), [4, 5, 6, 7])
        self.assertEqual(mean_absolute_errors(result)['Naive'], 1)


if __name__ == '__main__':
    unittest.main()
