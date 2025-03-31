from src.hVOS.camera import Camera
from src.hVOS.morphology import Morphology
from src.hVOS.cell import Cell
import unittest


""" Unit tests for methods of the Camera class. """
class TestCamera(unittest.TestCase):
    def setUp(self):
        self.cam = Camera()

    def load_test_cell(self):
        """ Create a mock cell for testing.
        Located at the origin
        has a soma, axon, apic, dend
        """

        test_axon = {'axon_0': 'tests/test_data/test_axon_0.dat'}
        test_apic = {'apic_0': 'tests/test_data/test_apic_0.dat'}
        test_dend = {'dend_0': 'tests/test_data/test_dend_0.dat'}
        test_soma = {'soma': 'tests/test_data/test_soma.dat'}

        cell = Cell("test_cell_id", 'test_me_type', 
                    test_axon, test_apic, 
                    test_dend, test_soma, 0, 0, 0,)
        return cell

    def load_test_morphology(self):
        """ Load a test morphology for the camera. """
        morphology_file = "tests/test_data/test_cell.nml"
        m = Morphology("test_me_type", morphology_file)
        return m

     

    def test_draw_weighted_line_0(self):
        """ Draw a weighted line with a 
        positive slope. """
        self.cam.draw_weighted_line(0, 0, 0, 1, 1, 1, 0.5, 0)

