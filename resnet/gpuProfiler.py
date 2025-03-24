import torch 
import subprocess

class GpuProfiler:
    """
        This profiler is based on the website https://pytorch.org/blog/understanding-gpu-memory-1/
    """

    def __init__(self,maxEntries:int,logFile:str,  enabled: bool):
        self.maxEntries = maxEntries
        self.logFile = logFile
        self.enabled = enabled
        self.started = False
        self.deviceProperties = torch.cuda.get_device_properties(device = torch.cuda.current_device()) 
        self.maxEntries = maxEntries
        self.numEntries = 0
        print("device Properties \n", self.deviceProperties)
        if(self.enabled):
            self.started = True
            torch.cuda.memory._record_memory_history(max_entries=maxEntries)

    def takeSnapshot(self):
        if(not self.enabled):
            return
        
        if(self.numEntries >= self.maxEntries):
            return
        
        try:
            torch.cuda.memory._dump_snapshot(f"{self.logFile}.pickle")
            self.numEntries+=1
        except Exception as e:
            print(f"Error taking snapshot: {e}")

    def startRecording(self):
        if(not self.enabled):
            return
        torch.cuda.memory._record_memory_history(enabled=True)
        self.started = True

    def stopRecording(self):
        if(not self.enabled):
            return
        torch.cuda.memory._record_memory_history(enabled=False)
        self.started = False
    
    def generateHtmlPlotFromSnapshot(self):
        if self.started:
            self.stopRecording()

        src = f"{self.logFile}.pickle"
        dest = f"{self.logFile}.html"

        cmd = "python3 /usr/local/lib/python3.10/dist-packages/torch/cuda/_memory_viz.py trace_plot {} -o {}".format(src, dest)
        subprocess.run(cmd, shell=True)