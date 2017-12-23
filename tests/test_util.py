import unittest
from json.decoder import JSONDecodeError

from forex.util import load_config, load_data, is_file_exists


class TestUtil(unittest.TestCase):

    def test_load_config(self):
        with self.assertRaises(FileNotFoundError):
            load_config("config.json")
        with self.assertRaises(JSONDecodeError):
            load_config("../data/DAT_MT_AUDUSD_M1_201711.csv")
        config = load_config("../config.json")
        self.assertTrue(len(config) > 0)

    def test_load_data(self):
        with self.assertRaises(FileNotFoundError):
            load_data("data.csv", ["col_1", "col_2", "col_3"])
        data = load_data("../data/DAT_MT_AUDUSD_M1_201711.csv",
                         ["date", "time", "open", "high", "low", "close", "volume"])
        self.assertTrue(len(data) > 0)

    def test_is_file_exists(self):
        self.assertFalse(is_file_exists("config.json"))
        self.assertTrue(is_file_exists("../config.json"))


if __name__ == '__main__':
    unittest.main()
