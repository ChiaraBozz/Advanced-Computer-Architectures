import subprocess
import re
import json

def run_c_program(program, a, b):
    try:
        result = subprocess.check_output([program, str(a), str(b)], universal_newlines=True)
        #return int(result.strip())
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


def refine_results(result):
    try:
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', result)
        if len(matches) >= 3:
            values = [float(match) for match in matches[-3:]] 
            return values
        else:
            print("Error: Failed to extract numeric values from C program output.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    input_tuple = [ (1024, 32), (1024, 256),
                    (2048, 32), (2048, 256),
                    (76744, 32), (76744, 256), 
                    (1048576, 32),(1048576, 64),(1048576, 128),(1048576, 256),(1048576, 512),(1048576, 1024),
                    (16777216, 32),(16777216, 64),(16777216, 128),(16777216, 256),(16777216, 512),(16777216, 1024),
                    (67108864, 32),(67108864, 64),(67108864, 128),(67108864, 256),(67108864, 512),(67108864, 1024)
                    #(268435456, 32),(268435456, 256),(268435456, 512),(268435456, 1024),
                    #(536870912, 32),(536870912, 256),(536870912, 512),(536870912, 1024)
                   #(402400000, 1024)
                   ]
    #input_tuple = [(16777216, 256),(16777216, 32),(76744, 32), (1024, 17), (3777, 32), (4048, 27), (1048576, 256), (33554432, 512)]
    programs = ["./reduce", "./reduce_without_shmem", "./reduce_shmem_less_div", "./reduce_task_par"]
    #programs = ["./reduce_task_par"]
    list_data = []
    
    for input in input_tuple:
        for program in programs:
            #N, BLOCK_SIZE = 16000, 32
            N, BLOCK_SIZE = input

            result = run_c_program(program, N, BLOCK_SIZE)
            #print(result)
            if result is None:
                print("Failed to run C program.")
                #exit(0)
                
            else:
                #print(f"Result from C program:\n{result}")

                res = refine_results(result)
                
                data = {
                    "program": program,
                    "N": N,
                    "BLOCK_SIZE": BLOCK_SIZE, 
                    "HostTime": res[0],
                    "KernelTime": res[1],
                    "SpeedUp": res[2]
                }
                list_data.append(data)
                print(data)

    json_file_path = "performance.json"

    with open(json_file_path, 'w') as json_file:
        json.dump(list_data, json_file, indent=2)

    print(f"Data has been written to {json_file_path}")

