from src.hVOS.camera import Camera


class SubConnMap:

    """ Given a Cell object and a filepath of a v7_batch1_0_0.run file,
        this class will create a dict recording the presynaptic cells,
        the postsynaptic cell compartment, and the list of locations
        of the synapses.
        This will be used in the Camera class to plot the synapses 
        if the option is enabled.
    """
    def __init__(self, run_filepath, post_cell_id):
        """
        Initialize the SubConnPlot class.

        Args:
            cell_id (int): The ID of the cell to plot.
            filepath (str): Path to the v7_batch1_0_0.run file.
        """
        
        self.run_filepath = run_filepath
        self.post_cell_id = post_cell_id
        self.subconn_map = {}
        self.get_subconn_map()
        

    def get_subconn_map(self):
        """
        Get the subconn map for the given cell ID.

        Returns:
            dict: A dictionary mapping presynaptic cells to synapse number
                  to postsynaptic cell compartment and location of the synapse.
        """
        with open(self.run_filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "Created connection" in line and f"postGid={self.post_cell_id}" in line:

                parts = line.split(' ')
                assert 'preGid' in parts[2], "Error: preGid not found in line."
                assert 'postGid' in parts[3], "Error: postGid not found in line."
                assert 'sec' in parts[4], "Error: sec not found in line."
                assert 'loc' in parts[5], "Error: loc not found in line."
                pre_cell_id = int(parts[2].split('=')[1])
                sec = parts[4].split('=')[1]
                loc = float(parts[5].split('=')[1])
                
                if sec not in self.subconn_map:
                    self.subconn_map[sec] = {}
                
                if pre_cell_id not in self.subconn_map[sec]:
                    self.subconn_map[sec][pre_cell_id] = []
                
                self.subconn_map[sec][pre_cell_id].append(loc)
        
