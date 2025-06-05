"""Utilities for expanding atomic coordinates via symmetry operations."""

import numpy as np
import spglib

def get_Atomic_Coordinates(positions, space_group_operations, atomic_labels, cell_params):
    # Apply space group symmetry operations to generate all atomic fractional coordinates
    all_positions = []
    all_labels = []
    for i, pos in enumerate(positions):
        label = atomic_labels[i]
        for rotation, translation in zip(space_group_operations['rotations'], space_group_operations['translations']):
            new_pos = np.dot(pos, np.transpose(rotation)) + translation
            all_positions.append(new_pos)
            all_labels.append(label)

    # Use cell parameters provided from the CIF file
    a, b, c, alpha, beta, gamma = cell_params

    # Instead of returning (label, [x, y, z]), return (label, x, y, z) directly
    atoms = [(label, pos[0], pos[1], pos[2]) for label, pos in zip(all_labels, all_positions)]

        # Define the dtype for the structured array
    atom_dtype = np.dtype([
        ('type', 'U10'),  # String type with length up to 10
        ('x', 'f8'),      # Float64 for x, y, z
        ('y', 'f8'),
        ('z', 'f8')
        
        ])

    # Create a structured array
    atoms_array = np.array(atoms, dtype=atom_dtype)

    # Now you can access the fields by name
    for i in range(len(atoms_array)):
        atom_type = atoms_array[i]['type']
        x = atoms_array[i]['x']
        y = atoms_array[i]['y']
        z = atoms_array[i]['z']

    return (a, b, c, alpha, beta, gamma), atoms

def write_xtl(lattice, positions, numbers, space_group_operations, atomic_labels, cell_params, filename="output.xtl"):
    (a, b, c, alpha, beta, gamma), atoms = get_Atomic_Coordinates(lattice, positions, numbers, space_group_operations, atomic_labels, cell_params)
    
    # Define the dtype for the structured array
    atom_dtype = np.dtype([
        ('type', 'U10'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('z', 'f8')
    ])

    # Now we can directly convert atoms into a structured array without extra formatting
    atoms_array = np.array(atoms, dtype=atom_dtype)
    
    # Write to the .xtl file
    with open(filename, "w") as f:
        f.write("TITLE Bi2 Se3\n")
        f.write("CELL\n")
        f.write(f"  {a:.6f}   {b:.6f}  {c:.6f}  {alpha:.6f}  {beta:.6f}  {gamma:.6f}\n")
        f.write("SYMMETRY NUMBER 1\n")
        f.write("SYMMETRY LABEL  P1\n")
        f.write("ATOMS\n")
        f.write("NAME         X           Y           Z\n")
        for atom in atoms_array:
            f.write("{:<12s}{:11.6f}{:11.6f}{:11.6f}\n".format(atom['type'], atom['x'], atom['y'], atom['z']))
        f.write("EOF\n")

