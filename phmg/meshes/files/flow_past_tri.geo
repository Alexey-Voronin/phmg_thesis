// Define parameters for the geometry
Lx = 5.0;              // Length of the domain in the x-direction
Ly = 3.0;              // Length of the domain in the y-direction
R  = 0.5;              // Radius of the cylinder
cl1 = 0.2;            // Characteristic length (mesh size) near the cylinder
cl2 = 0.8;             // Characteristic length (mesh size) away from the cylinder

// Points for the rectangular domain
Point(1) = {0, 0, 0, cl2};
Point(2) = {Lx, 0, 0, cl2};
Point(3) = {Lx, Ly, 0, cl2};
Point(4) = {0, Ly, 0, cl2};

// Lines for the rectangular domain
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define the center of the cylinder
Point(5) = {Lx/2, Ly/2, 0, cl1};

// Create points on the circle boundary using the center and the radius
Point(6) = {Lx/2 + R, Ly/2, 0, cl1};
Point(7) = {Lx/2, Ly/2 + R, 0, cl1};
Point(8) = {Lx/2 - R, Ly/2, 0, cl1};
Point(9) = {Lx/2, Ly/2 - R, 0, cl1};

// Create circle segments for the cylinder boundary
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Close the cylinder boundary
Line Loop(5) = {5, 6, 7, 8};

// Close the rectangular boundary and subtract the cylinder loop
Line Loop(6) = {1, 2, 3, 4};
Plane Surface(1) = {6, 5};

// Physical Groups (Optional: for marking boundary conditions or for exporting purposes)
Physical Line("Inlet") = {1};
Physical Line("Outlet") = {3};
Physical Line("Walls") = {2, 4};
Physical Line("Cylinder") = {5, 6, 7, 8};
Physical Surface("Fluid") = {1};

// Generate 2D mesh
Mesh 2;

