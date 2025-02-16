#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>

#include "kaizen.h"

// IR Node representing a single operation.
struct IrNode {
    std::string name;
    std::string op;                        // Operation type: "Constant" or "Add"
    std::vector<std::string> operandNames; // For "Add", stores operand names
    int value;                             // Used if op == "Constant"
};

// IR Module containing a list of nodes and a lookup table.
struct IrModule {
    std::vector<IrNode*> nodes;
    std::map<std::string, IrNode*> nodeMap;
};

// Utility function to print the IR module.
void printModule(IrModule* module) {
    for (auto node : module->nodes) {
        if (node->op == "Constant") {
            zen::log("Constant", node->name, "=", node->value);
        }
        else if (node->op == "Add") {
            zen::log("Add", node->name, "=", node->operandNames[0], "+", node->operandNames[1]);
        }
    }
}

// Function to parse a minimal IR module from an input file.
// The expected syntax for each line is:
// Constant <name> = <value>
// Add <name> = <operand1> + <operand2>
IrModule* parseModule(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        zen::log("Error opening file ", zen::quote(filename));
        return nullptr;
    }
    IrModule* module = new IrModule();
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string op;
        iss >> op;         // read the operation type
        IrNode* node = new IrNode();
        node->op = op;
        iss >> node->name; // read the node name
        // Skip the '=' token.
        std::string eq;
        iss >> eq;
        if (op == "Constant") {
            int val;
            iss >> val;
            node->value = val;
        } else if (op == "Add") {
            std::string operand1, plus, operand2;
            iss >> operand1 >> plus >> operand2;
            node->operandNames.push_back(operand1);
            node->operandNames.push_back(operand2);
        } else {
            zen::log("Unknown operation type: ", zen::quote(op));
        }
        module->nodes.push_back(node);
        module->nodeMap[node->name] = node;
    }
    return module;
}

// Error predicate: In this minimal example, the property is preserved if the module contains at least one "Add" node
// whose operands are defined.
bool checkPrimaryPredicate(IrModule* module) {
    for (auto node : module->nodes) {
        if (node->op == "Add") {
            if (module->nodeMap.contains(node->operandNames[0]) &&
                module->nodeMap.contains(node->operandNames[1]))
                return true;
        }
    }
    return false;
}

// Reduction function: attempts to remove a node (non-critical node, such as a "Constant") and checks whether the property holds.
// Returns true if a reduction was applied.
bool reduceModule(IrModule* module) {
    // Try removing non-"Add" nodes first to preserve the property.
    for (int i : zen::in(module->nodes.size())) {
        IrNode* node = module->nodes[i];
        if (node->op != "Add") {
            // Temporarily remove the node.
            module->nodes.erase(module->nodes.begin() + i);
            module->nodeMap.erase(node->name);
            zen::print("Reduction applied, ");

            // Check if the property is still preserved.
            if (checkPrimaryPredicate(module)) {
                zen::log(zen::color::green("predicate holds"), "after removing node", zen::quote(node->name));
                delete node;
                return true;
            }
            else {
                // Revert removal if property is lost.
                module->nodes.insert(module->nodes.begin() + i, node);
                module->nodeMap[node->name] = node;
            }
        }
    }
    return false; // No valid reduction could be applied.
}

int main(int argc, char* argv[]) {
    zen::cmd_args args(argv, argc);
    if (argc < 2) {
        zen::log("Usage: irreduce <input_file>");
        return 1;
    }

    // Parse the input IR module.
    IrModule* module = parseModule(args.arg_at(1));
    if (!module) return 1;

    zen::log("Original Module:");
    printModule(module);

    // Iterative reduction: apply reductions until no further change is possible.
    int reductionCount = 0;
    while (reduceModule(module)) {
        reductionCount++;
        // zen::log("\nAfter ", reductionCount, " reduction(s):\n");
        // printModule(module);
    }

    zen::log("final module after", reductionCount, "reductions:");
    printModule(module);

    // Cleanup: deallocate memory.
    for (auto node : module->nodes) {
        delete node;
    }
    delete module;
}