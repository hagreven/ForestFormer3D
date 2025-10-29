import laspy

def read_las(filename):
    las = laspy.read(filename)
    return {
        "x": las.x,
        "y": las.y,
        "z": las.z,
        # Add more fields as needed
    }
    