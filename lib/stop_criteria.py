import unittest

from lib.experiment import stop_criteria


class StopCriteria(unittest.TestCase):
    def test_stop_criteria(self):
        shouldnt_stop = [90, 90, 90, 90, 100, 100]
        shouldnt_stop2 = [90, 90, 0]
        should_stop = [90, 100, 100, 100, 100, 100]

        self.assertEqual(stop_criteria(shouldnt_stop), False)
        self.assertEqual(stop_criteria(shouldnt_stop2), False)
        self.assertEqual(stop_criteria(should_stop), True)


if __name__ == '__main__':
    unittest.main()
