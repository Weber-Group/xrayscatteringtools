def read_xyz(filename):
    """Read an XYZ file and return atoms and coordinates."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    atoms = []
    coords = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(element)
        coords.append((x, y, z))
    
    return num_atoms, comment, atoms, coords