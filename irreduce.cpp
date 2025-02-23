#include <functional>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>

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

// Converts an IR module to a string representation.
std::string to_string(IrModule* module) {
    std::stringstream result;
    for (auto node : module->nodes) {
        if (node->op == "Constant") {
            result << "Constant " << node->name << " = " << node->value << '\n';
        } else if (node->op == "Add") {
            result << "Add " << node->name << " = " << node->operandNames[0] << " + " << node->operandNames[1] << '\n';
        }
    }
    return result.str();
}

// Function to parse a minimal IR module from an input file.
// The expected syntax for each line is:
//     Constant <name> = <value>
//     Add <name> = <operand1> + <operand2>
IrModule* parseIR(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::string msg = "Unable to open the file " + std::string(zen::color::red(zen::quote(filename)));
        throw std::runtime_error(msg);
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

// Primary predicate: the IR invariant holds if the module contains
// at least one "Add" node whose operands are defined.
bool predicatePrimary(IrModule* module) {
    for (auto node : module->nodes) {
        if (node->op == "Add") {
            if (module->nodeMap.contains(node->operandNames[0]) &&
                module->nodeMap.contains(node->operandNames[1]))
                return true;
        }
    }
    return false;
}

// Helper function: creates a deep copy of the module.
IrModule* cloneModule(IrModule* module) {
    IrModule* newModule = new IrModule();
    for (auto node : module->nodes) {
        IrNode* newNode = new IrNode();
        newNode->name = node->name;
        newNode->op = node->op;
        newNode->operandNames = node->operandNames;
        newNode->value = node->value;
        newModule->nodes.push_back(newNode);
        newModule->nodeMap[newNode->name] = newNode;
    }
    return newModule;
}

// Helper function: deallocates the module.
void freeModule(IrModule* module) {
    for (auto node : module->nodes) {
        delete node;
    }
    delete module;
}

// Type alias for a transformation pass function.
using Pass = std::function<bool(IrModule*)>;

// Pass registry: returns a reference to a static vector of passes.
std::vector<Pass>& getPassRegistry() {
    static std::vector<Pass> registry;
    return registry;
}

// Function to register a new pass.
void registerPass(Pass pass) {
    getPassRegistry().push_back(pass);
}

// Type alias for a predicate function.
using Predicate = std::function<bool(IrModule*)>;

// Predicate registry: returns a reference to a static vector of predicates.
std::vector<Predicate>& getPredicateRegistry() {
    static std::vector<Predicate> registry;
    return registry;
}

// Function to register a new predicate.
void registerPredicate(Predicate pred) {
    getPredicateRegistry().push_back(pred);
}

// A reduction pass that attempts to remove a non-critical node ("Constant").
// Returns true if a node was successfully removed.
bool passReduction(IrModule* module) {
    for (int i : zen::in(module->nodes.size())) {
        IrNode* node = module->nodes[i];
        if (node->op != "Add") {
            // Temporarily remove the node.
            module->nodes.erase(module->nodes.begin() + i);
            module->nodeMap.erase(node->name);
            zen::log(std::string(zen::color::magenta(__func__)) + ":", "removed node", zen::quote(node->name));
            delete node;
            return true;
        }
    }
    zen::log(std::string(zen::color::magenta(__func__)) + ":", "no further reduction possible.");
    return false;
}

// Pass that removes all unused constants.
bool passRemoveUnusedConstants(IrModule* module) {
    bool removed = false;
    std::set<std::string> used_names;
    // Collect all operand names from "Add" nodes.
    for (auto node : module->nodes) {
        if (node->op == "Add") {
            for (const std::string& operand : node->operandNames) {
                used_names.insert(operand);
            }
        }
    }
    // Remove nodes in reverse order to avoid invalidating iterators.
    for (int i = module->nodes.size() - 1; i >= 0; --i) {
        IrNode* node = module->nodes[i];
        if (node->op == "Constant" && used_names.find(node->name) == used_names.end()) {
            module->nodes.erase(module->nodes.begin() + i);
            module->nodeMap.erase(node->name);
            zen::log(std::string(zen::color::magenta(__func__)) + ":", "removed node", zen::quote(node->name));
            delete node;
            removed = true;
        }
    }
    return removed;
}

namespace NAME::ARG {
    static const std::string input_file = "--input_file"; // Path to the input file containing the IR module.
};

int main(int argc, char* argv[]) try {
    // Parse the command line arguments.
    zen::cmd_args args(argv, argc);
    
    // Check if the required argument(s) are present.
    if (!args.accept(NAME::ARG::input_file).is_present()) {
        throw std::invalid_argument("Missing required argument: " + NAME::ARG::input_file);
    }

    auto input_file = args.get_options(NAME::ARG::input_file)[0];

    // Parse the input IR module.
    IrModule* module = parseIR(input_file);

    zen::log("Original Module:\n");
    zen::log(module);

    // Register transformation passes.
    registerPass(passReduction);
    registerPass(passRemoveUnusedConstants);
    
    // Register predicates.
    registerPredicate(predicatePrimary);

    // Run registered passes iteratively until no changes occur.
    int pass_count = 0;
    bool pass_applied;
    do { // Apply each pass in the registry.
        pass_applied = false;
        for (auto& pass : getPassRegistry()) {
            // Backup the current state of the module.
            IrModule* backup = cloneModule(module);
            if (pass(module)) {
                // After applying the pass, verify all registered predicates.
                bool invariant = true;
                for (auto& pred : getPredicateRegistry()) {
                    if (!pred(module)) {
                        invariant = false;
                        break;
                    }
                }
                if (!invariant) {
                    zen::log(zen::color::yellow("A predicate failed after most recent pass; reverting it..."));
                    freeModule(module);
                    module = backup; // Restore from backup.
                } else {
                    // The pass is valid.
                    pass_applied = true;
                    pass_count++;
                    freeModule(backup);
                }
            } else {
                freeModule(backup);
            }
        }
    } while (pass_applied);
    
    zen::log("Final module after", pass_count, "reductions:\n");
    zen::log(module);
    
    // Cleanup: deallocate memory. This will be rewritten later
    // with proper resource management after this POC phase.
    freeModule(module);
} catch (const std::exception& e) {
    zen::log(zen::color::red("ERROR:"), e.what());
    return 1;
}
