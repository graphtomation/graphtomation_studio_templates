import os
import sys
import unittest
import importlib.util
from django.test import TestCase

def discover_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    base_dir = os.path.dirname(__file__)
    
    # Walk through subdirectories of the app (skipping the root)
    for root, dirs, files in os.walk(base_dir):
        # Skip the root directory (which is this aggregator file)
        if os.path.abspath(root) == os.path.abspath(base_dir):
            continue
        
        if 'tests.py' in files:
            test_file = os.path.join(root, 'tests.py')
            # Convert the file path to a module name. For example:
            # "subfolder/tests.py" becomes "subfolder.tests"
            rel_path = os.path.relpath(test_file, base_dir)
            module_name = rel_path.replace(os.path.sep, '.')[:-3]  # remove ".py"
            
            # Dynamically import the tests.py module
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for any classes that are a subclass of django.test.TestCase
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (
                    isinstance(attribute, type) and 
                    issubclass(attribute, TestCase) and 
                    attribute is not TestCase
                ):
                    suite.addTests(loader.loadTestsFromTestCase(attribute))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    suite = discover_tests()
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
