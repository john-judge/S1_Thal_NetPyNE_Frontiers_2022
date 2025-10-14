import re


class Morphology:
    def __init__(self, me_type, me_type_file_path):
        self.me_type = me_type
        self.structure_data = None
        self.me_type_file_path = me_type_file_path
        self.load_morphology(me_type_file_path)

    def load_morphology(self, file_path):
        ''' Load morphology data from a file. Each segment looks like:
            <morphology id="morphology">
                <segment id="0" name="Seg0_soma_0">
                    <proximal y="6.274640e+00" x="-3.084520e+00" z="3.124610e-01" diameter="1.466230e+00"/>
                    <distal y="5.640320e+00" x="-2.799270e+00" z="2.871150e-01" diameter="2.648440e+00"/>
                </segment>
             
               '''

        # Load the morphology data from the file
        morphology_data = None
        with open(file_path, 'r') as f:
            morphology_data = f.readlines()

        # Parse the morphology data based on name='Seg#_section_3 and x='' y='' z='' diameter=''
        # Store the parsed data in the structure_data attribute
        # The structure_data should look like:
        # {
        #     'soma': {
        #         '0': {
        #             'proximal': {
        #                 'x': -3.084520e+00,
        #                 'y': 6.274640e+00,
        #                 'z': 3.124610e-01,
        #                 'diameter': 1.466230e+00
        #             },
        #             'distal': {
        #                 'x': -2.799270e+00,
        #                 'y': 5.640320e+00,
        #                 'z': 2.871150e-01,
        #                 'diameter': 2.648440e+00
        #             }
        #         }
        #     }
        # }
        self.structure_data = {}
        
        for line in morphology_data:
            # do not rely on the order of the attributes, 
            # use e.g. re.search(r'x="([^"]+)"', line).group(1)

            if 'name="Seg' in line:
                segment_id = int(re.search(r'id="([^"]+)"', line).group(1))
                segment_name = re.search(r'name="([^"]+)"', line).group(1)
                segment_name = segment_name.split('_')
                segment_type = "_".join(segment_name[1:])

                if segment_type not in self.structure_data:
                    self.structure_data[segment_type] = {}
                if segment_id not in self.structure_data[segment_type]:
                    self.structure_data[segment_type][segment_id] = {}
            if 'proximal' in line or 'distal' in line:
                segment_end = 'proximal' if 'proximal' in line else 'distal'
                try:
                    self.structure_data[segment_type][segment_id][segment_end] = {
                        'x': float(re.search(r'x="([^"]+)"', line).group(1)),
                        'y': float(re.search(r'y="([^"]+)"', line).group(1)),
                        'z': float(re.search(r'z="([^"]+)"', line).group(1)),
                        'diameter': 
                            float(re.search(r'diameter="([^"]+)"', line).group(1))
                    }
                except AttributeError:
                    if 'translationStart' in line:
                        pass
                    else:
                        print(f"Error parsing {line}")
                        print(f"segment_type: {segment_type}, segment_id: {segment_id}, segment_end: {segment_end}")
                        raise AttributeError

    def get_structure(self):
        return self.structure_data
    
    def get_compartment_id_list(self):
        return list(self.structure_data.keys())
    
    def does_cell_match_morphology(self, cell):
        # Check if the cell morphology matches the morphology structure        
        if cell.get_me_type().split("_barrel")[0] != self.me_type:
            return False
        
        # then check if the cell's morphology structure matches the
        #  structure_data of this morphology (compartment lists should match)
        morph_compartment_list = self.get_compartment_id_list()
        cell_compartment_list = cell.get_list_compartment_ids()
        if len(morph_compartment_list) != len(cell_compartment_list):
            return False
        
        cell_compart_dict = {comp_id.replace("V", "").replace("soma_0","soma"):True 
                             for comp_id in cell_compartment_list}
        
        print("morph_compartment_list:", morph_compartment_list)
        print("cell_compartment_list:", cell_compartment_list)
        return all([comp_id in cell_compart_dict for 
                            comp_id in morph_compartment_list if 'soma' not in comp_id])

    def manipulate_structure(self, label, transformation):
        # Apply a transformation to the morphology structure
        # adds a key LABEL to each segment in self.structure_data[label] with the value transformation
        pass

    def __repr__(self):
        return f"Morphology(me_type={self.me_type})"