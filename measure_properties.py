import numpy as np

class TraceProperties:
    def __init__(self, pro_data, start, width, int_pts, per_amp=0.5, rli=1.0):
        """
        Parameters
        ----------
        pro_data : np.ndarray
            1-D numpy array of processed data (like `proData`).
        start : int
            Start index of the analysis window.
        width : int
            Width of the analysis window.
        int_pts : float
            Integration points (sampling interval).
        per_amp : float
            Proportion of amplitude for half-amp latency (default 0.5).
        rli : float
            RLI baseline value for this trace (default: 1.0).
        """
        self.pro_data = pro_data
        self.start = start
        self.width = width

        # truncate width if start + width exceeds data length
        if self.start + self.width > len(self.pro_data):
            self.width = len(self.pro_data) - self.start - 1
        self.int_pts = int_pts
        self.per_amp = per_amp
        self.rli = rli

        # Outputs
        self.max_amp = None
        self.max_amp_latency = None
        self.half_amp_latency = None
        self.max_amp_latency_pt = None
        self.half_amp_latency_decay = None
        self.half_width = None
        

        # Spike properties
        self.spike_start = None
        self.spike_end = None

        # Run calculations
        self.measure_properties()

    def measure_properties(self):
        num_pts = len(self.pro_data)

        #-------------------------------------------------------
        # 2. Max Amp
        # 3. Max Amp Latency
        self.max_amp = 0.0
        self.max_amp_latency = self.start + self.width

        for i in range(self.start, min(self.start + self.width + 1, num_pts)):
            if self.pro_data[i] > self.max_amp:
                self.max_amp = self.pro_data[i]
                self.max_amp_latency = i

        #-------------------------------------------------------
        # 5. Half Amp Latency
        if self.max_amp == 0:
            self.half_amp_latency = self.start
            return

        half_amp = self.max_amp * self.per_amp
        self.half_amp_latency = self.start

        for i in range(self.start, self.max_amp_latency + 1):
            if self.pro_data[i] == half_amp:
                self.half_amp_latency = float(i)
                break
            elif self.pro_data[i] > half_amp:
                if i == self.start:
                    self.half_amp_latency = float(i)
                    break
                denom = self.pro_data[i] - self.pro_data[i-1]
                if denom == 0:
                    self.half_amp_latency = float(i)
                else:
                    self.half_amp_latency = float(i) - 1 + (half_amp - self.pro_data[i-1]) / (self.pro_data[i] - self.pro_data[i-1])
                break

        # calculate time to reach halfAmp in the decay
        for i in range(self.max_amp_latency + 1, self.start + self.width + 1):
            if self.pro_data[i] < half_amp:
                if i == self.max_amp_latency + 1:
                    self.half_amp_latency_decay = float(i)
                    break
                denom = self.pro_data[i-1] - self.pro_data[i]
                if denom == 0:
                    self.half_amp_latency_decay = float(i)
                else:
                    self.half_amp_latency_decay = float(i) - 1 + (self.pro_data[i-1] - half_amp) / (self.pro_data[i-1] - self.pro_data[i])
                break
        if self.half_amp_latency_decay is None:
            self.half_amp_latency_decay = float(self.start)
        self.half_width = self.half_amp_latency_decay - self.half_amp_latency
        if self.half_width is None or self.half_width < 0:
            self.half_width = 0
        
        #-------------------------------------------------------
        self.max_amp_latency_pt = int(self.max_amp_latency)
        self.max_amp_latency *= self.int_pts
        self.half_amp_latency *= self.int_pts
        self.half_width *= self.int_pts


    def get_max_amp(self):
        return self.max_amp

    def get_max_amp_latency(self):
        return self.max_amp_latency

    def get_half_amp_latency(self):
        return self.half_amp_latency

    def get_spike_start(self):
        return self.spike_start

    def get_spike_end(self):
        return self.spike_end
    
    def get_half_amp_latency_decay(self):
        return self.half_amp_latency_decay
    
    def get_half_width(self):
        return self.half_width

