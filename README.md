# IRReduce

IRReduce is a tool for reducing IRs to a minimal form that preserves user-provided invariants.

## Build & Run on Windows & Linux (including [WSL](https://learn.microsoft.com/en-us/windows/wsl/install))

1. **Clone this repo:** Create a local copy of this repository on your machine.

2. **Navigate into the cloned repo directory:** `cd` into the directory (where your project's `CMakeLists.txt` file is located). Replace `path/to/your/cloned/dir` with the actual path:

   ```bash
   cd path/to/your/cloned/dir
   ```

3. **Create a Build Directory:** It's a good practice to create a separate directory for the build files to keep them isolated from the source files. Inside your project directory, run:

   ```bash
   mkdir build
   cd build
   ```

4. **Configure the Project with CMake:** Now, from inside the `build` directory, run CMake to configure the project. This will read the `CMakeLists.txt` file and generate the necessary Makefiles for building the project. The `..` at the end of the command tells CMake to look in the parent directory for the `CMakeLists.txt` file:

   ```bash
   cmake ..
   ```
**NOTE:** This will only work smoothly either within a Linux environment (including [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)) or from within a Windows terminal that's integrated
into Visual Studio (not Visual Studio Code), where the integrated terminal automatically sets up all the necessary environment variables (pointing to the MSVC compiler `cl.exe`, etc.) that are
automatically configured when you run Visual Studio's own developer command prompt.

If you run this from a Windows terminal inside another IDE like Visual Studio Code, you will probably get an error that looks like this:

```bash
-- Building for: NMake Makefiles
-- The C compiler identification is unknown
-- The CXX compiler identification is unknown
CMake Error at CMakeLists.txt:11 (project):
  The CMAKE_C_COMPILER:

    cl

  is not a full path and was not found in the PATH.
```
If you want to develop using MSVC compiler from Visual Studio Code, there are various ways of setting that up, please see online.

   If you need to specify a particular version of g++ or any other build options, you can do so with additional arguments. For example, to set the C++ compiler to g++, you could use:

   ```bash
   cmake -DCMAKE_CXX_COMPILER=g++ ..
   ```

5. **Build the Project:** After configuring, you can compile the project with:

   ```bash
   cmake --build .
   ```
   
   On Linux, you can also call ```make``` directly (does the same thing).

6. **Run the Executable:** If the build is successful, you can run the resulting executable from the build directory.
  
   Linux:

   ```bash
   ./irreduce <ir_file>
   ```
   Windows:
   ```bash
   .\irreduce.exe <ir_file>
   ```
   For example, using the toy (demo) IR files in the `ir` directory:
   ```bash
   ./irreduce ../ir/ir_1.txt --pass_unusedconstants
   ```
   Will produce an output like the following:
   ```
   Original Module:

   Constant x = 0
   Constant a = 5
   Constant b = 3
   Constant c = 7
   Add d = a + b
   Add e = d + c
   
   Applying passes: those specified explicitly.
   passRemoveUnusedConstants: removed node "x"
   Final module after 1 reductions:
   
   Constant a = 5
   Constant b = 3
   Constant c = 7
   Add d = a + b
   Add e = d + c
   ```
