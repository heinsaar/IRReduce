#include <functional>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <regex>

#include "kaizen.h"

namespace NAME::ARG {
    static const std::string input_file = "--input_file"; // Path to the input file containing the IR module.
    static const std::string invariants = "--invariants"; // Path to the external invariants script.
    static const std::string output_file = "--output_ir_file";

    // Passes
    static const std::string pass_noncriticals    = "--pass_noncriticals";    // Removes non-critical nodes.
    static const std::string pass_unusedconstants = "--pass_unusedconstants"; // Removes unused constants.
};

namespace NAME::OP {
    static const std::string constant = "constant";
    static const std::string add      = "add";
};

struct IrNode {
    std::string name;                      // node symbol
    std::string op;                        // "constant", "add", ...
    std::string type;                      // tensor type literal
    std::vector<std::string> operandNames; // user-defined names of operands
    std::string value;                     // raw constant payload
};

// IR Module containing a list of nodes and a lookup table.
struct IrModule {
    std::vector<IrNode*> nodes;
    std::map<std::string, IrNode*> nodeMap;
};

using OpHandler = std::function<std::string(const IrNode*)>;

std::map<std::string, OpHandler>& getOpRegistry() {
    static std::map<std::string, OpHandler> registry;
    return registry;
}

// Converts an IR module to a string representation (HLO style).
std::string to_string(IrModule* module) {
    std::stringstream result;

    result << "HloModule todo_replace_me_with_original_name\n\n";
    result << "ENTRY main {\n";

    for (auto node : module->nodes) {
        auto& registry = getOpRegistry();
        if (auto it = registry.find(node->op); it != registry.end()) {
            result << "  " << it->second(node) << "\n";
        } else {
            zen::log(zen::color::yellow("Unknown op type in printer:"), zen::quote(node->op));
        }
    }

    result << "  ROOT root = (";
    for (size_t i = 0; i < module->nodes.size(); ++i) {
        result << module->nodes[i]->type << ' ' << module->nodes[i]->name;
        if (i + 1 < module->nodes.size()) result << ", ";
    }

    result << ") tuple(";
    for (size_t i = 0; i < module->nodes.size(); ++i) {
        result << module->nodes[i]->name;
        if (i + 1 < module->nodes.size())
            result << ", ";
    }
    result << ")\n}\n";

    return result.str();
}

std::string to_string(IrNode* node) {
    return zen::to_string("Node:", node->name, node->op, node->operandNames);
}

void registerOpHandler(const std::string& op, OpHandler handler) {
    getOpRegistry()[op] = handler;
}

// ──────────────────────────────────────────────────────────────────────────────
//  HLO-aware parser (uses zen::string throughout)
//  Accepts a subset that is sufficient for your example:
//
//      <name> = s32[] constant(<int>)
//      <name> = s32[] add(<op1>, <op2>)
//
//  Everything else (HloModule header, ENTRY line, braces, ROOT tuple, layout
//  annotations, comments, blank lines) is ignored.
// ──────────────────────────────────────────────────────────────────────────────
IrModule* parseIR(const std::string& filename)
{
    std::ifstream ir_file(filename);
    if (!ir_file)
        throw std::runtime_error("Unable to open " + filename);

    IrModule* module = new IrModule();

    // <name> = <type> constant(<payload>)
    const std::regex re_const(R"(^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z0-9_\[\],]+)\s*constant\(\s*([^)]+)\s*\)\s*$)");

    // <name> = <type> add(<lhs>, <rhs>)
    const std::regex re_add(R"(^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z0-9_\[\],]+)\s*add\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)\s*$)");

    zen::string line;
    while (std::getline(ir_file, line)) {
        line.trim();                   // drop leading / trailing white–space
        if (line.is_empty()) continue;
        if (line[0] == '#')  continue; // ignore comments
        if (line.contains("HloModule")
        || line.contains("ENTRY")
        || line.contains("{")
        || line.contains("}")
        || line.contains("ROOT"))
            continue; // skip structural noise

        std::smatch m;
        if (std::regex_match(line, m, re_const)) {
            auto* n = new IrNode{};
            n->name = m[1];
            n->type = m[2];
            n->op = NAME::OP::constant;
            n->value = m[3];
            module->nodes.push_back(n);
            module->nodeMap[n->name] = n;
            continue;
        }
        if (std::regex_match(line, m, re_add)) {
            auto* n = new IrNode{};
            n->name = m[1];
            n->type = m[2];
            n->op = NAME::OP::add;
            n->operandNames = { m[3], m[4] };
            module->nodes.push_back(n);
            module->nodeMap[n->name] = n;
            continue;
        }

        // Non-empty but still unrecognised: warn, keep going
        zen::log(zen::color::yellow("HLO parser: ignored unsupported or unfamiliar line:"), zen::quote(line));
    }
    return module;
}

// Primary invariant: the IR invariant holds if the module contains
// at least one "add" node whose operands are defined.
bool invariantAddPresent(IrModule* module) {
    for (auto node : module->nodes) {
        if (node->op == NAME::OP::add) {
            if (module->nodeMap.contains(node->operandNames[0]) &&
                module->nodeMap.contains(node->operandNames[1]))
                return true;
        }
    }
    return false;
}

// Helper function: creates a deep copy of the module.
IrModule* cloneModule(IrModule* module) {
    IrModule* new_module = new IrModule();
    for (auto node : module->nodes) {
        IrNode* new_node = new IrNode();
        new_node->name = node->name;
        new_node->op = node->op;
        new_node->operandNames = node->operandNames;
        new_node->type = node->type;
        new_node->value = node->value;
        new_module->nodes.push_back(new_node);
        new_module->nodeMap[new_node->name] = new_node;
    }
    return new_module;
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

// Type alias for a invariant function.
using Invariant = std::function<bool(IrModule*)>;

// Invariant registry: returns a reference to a static vector of invariants.
std::vector<Invariant>& getInvariantRegistry() {
    static std::vector<Invariant> registry;
    return registry;
}

// Function to register a new invariant.
void registerInvariant(Invariant inv) {
    getInvariantRegistry().push_back(inv);
}

// A reduction pass that attempts to remove a non-critical node ("constant").
// Returns true if a node was successfully removed.
bool passRemoveNoncriticals(IrModule* module) {
    for (int i : zen::in(module->nodes.size())) {
        IrNode* node = module->nodes[i];
        if (node->op != NAME::OP::add) {
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
    // Collect all operand names from "add" nodes.
    for (auto node : module->nodes) {
        if (node->op == NAME::OP::add) {
            for (const std::string& operand : node->operandNames) {
                used_names.insert(operand);
            }
        }
    }
    // Remove nodes in reverse order to avoid invalidating iterators.
    for (int i = module->nodes.size() - 1; i >= 0; --i) {
        IrNode* node = module->nodes[i];
        if (node->op == NAME::OP::constant && used_names.find(node->name) == used_names.end()) {
            module->nodes.erase(module->nodes.begin() + i);
            module->nodeMap.erase(node->name);
            zen::log(std::string(zen::color::magenta(__func__)) + ":", "removed node", zen::quote(node->name));
            delete node;
            removed = true;
        }
    }
    return removed;
}

const char* get_default_input_file_path() {
#ifdef _WIN32
    return "../../../ir/hlo_1.ir";
#else
    return "../ir/hlo_1.ir";
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) try {
    zen::cmd_args args(argv, argc);
    zen::log("Running IRReduce with command: ", args.original_command());

    std::string input_file_path;

    if (args.accept(NAME::ARG::input_file).is_present()) {
        auto input_options = args.get_options(NAME::ARG::input_file);
        if (input_options.empty()) {
            throw std::runtime_error("No input file specified for argument: " + NAME::ARG::input_file);
        }
        input_file_path = input_options[0];
    } else {
    #ifndef NDEBUG
        input_file_path = get_default_input_file_path(); // no input file specified, use default for debugging
        zen::log("No input file specified, using default for debugging:", zen::color::yellow(zen::quote(input_file_path)));
    #else
        throw std::invalid_argument("Missing required argument(s): input file path. Specify it implicitly "
            "by providing it as the only argument, or explicitly with: " + NAME::ARG::input_file + " <path>");
    #endif
    }

    /* ------------------------------------------------------------------------
    Output-file handling  (–‐output_ir_file)
    ------------------------------------------------------------------------ */
    std::string output_file_path;

    if (args.accept(NAME::ARG::output_file).is_present()) {
        // the flag was given
        auto out_opts = args.get_options(NAME::ARG::output_file);
        if (out_opts.empty())
            throw std::runtime_error("Flag given with no path: " + NAME::ARG::output_file);
        output_file_path = out_opts[0];
    } else {
        // derive default: ./output/<basename>.ir
        zen::string base = input_file_path;
        size_t slash = base.find_last_of("/\\");
        if (slash != zen::string::npos)
            base = base.substr(slash + 1); 
        size_t dot = base.rfind('.');
        if (dot != zen::string::npos) 
            base = base.substr(0, dot);                         // drop extension
        output_file_path = "output/" + base + ".ir";
    }

    zen::log("Output IR file will be:", zen::quote(output_file_path));

    
    registerOpHandler(NAME::OP::constant, [](const IrNode* n) {
        return n->name + " = " + n->type + " constant(" + n->value + ")";
    });

    registerOpHandler(NAME::OP::add, [](const IrNode* n) {
        return n->name + " = " + n->type + " add(" + n->operandNames[0] + ", " + n->operandNames[1] + ")";
    });

    // Parse the input IR module.
    IrModule* module = parseIR(input_file_path);

    zen::log("Original Module:\n");
    zen::log(module);

    // Check for individual passes.
    bool pass_noncriticals    = args.accept(NAME::ARG::pass_noncriticals).is_present();
    bool pass_unusedconstants = args.accept(NAME::ARG::pass_unusedconstants).is_present();

    // Determine if any passes are explicitly specified.
    bool has_explicit_passes = pass_noncriticals || pass_unusedconstants;

    // Apply all passes if none are explicitly specified.
    bool apply_all_passes = !has_explicit_passes;

    zen::log("----------------------------");
    zen::log("Preparing reduction with the following configuration:");
    zen::log("Passes:", apply_all_passes ? "all" : "those specified explicitly.");

    // Register transformation passes based on the input flags.
    if (apply_all_passes || pass_noncriticals)
        registerPass(passRemoveNoncriticals);

    if (apply_all_passes || pass_unusedconstants)
        registerPass(passRemoveUnusedConstants);

    // If the user provides an external invariants script, register its invariant.
    if (args.accept(NAME::ARG::invariants).is_present()) {
        // Manually split --key=value arguments for Zen compatibility
        zen::vector<zen::string> rewritten;
        for (int i = 1; i < argc; ++i) {
            zen::string arg = argv[i];
            auto pos = arg.find('=');
            if (pos != zen::string::npos) {
                zen::string key = arg.substr(0, pos);
                zen::string val = arg.substr(pos + 1);
                rewritten.push_back(key);
                rewritten.push_back(val);
            } else {
                rewritten.push_back(arg);
            }
        }

        zen::cmd_args args((char**)rewritten.data(), (int)rewritten.size());

        auto invariants_script = args.get_options(NAME::ARG::invariants)[0];
        zen::log("Using external invariants script:", zen::quote(invariants_script));
        registerInvariant([invariants_script](IrModule* module) -> bool {
            // Write the IR module to a temporary file.
            char tmp_filename[L_tmpnam];
            tmpnam(tmp_filename); // Note: use tmpnam for now, but it's not secure. Will be replaced.
            std::ofstream tmp_file(tmp_filename);
            if (!tmp_file) {
                throw std::runtime_error("Failed to create temporary file for invariants check.");
            }
            tmp_file << to_string(module);
            tmp_file.close();
            // Execute the user-provided invariants script.
            std::string command = "sh " + invariants_script + " " + tmp_filename;
            zen::log("Executing command:", zen::quote(command));
            int ret = system(command.c_str());
            std::remove(tmp_filename);
            return (ret == 0);
        });
    } else {
        zen::log("Invariants: default (no invariant script file specified, using default invariants for debugging).");
        // Register invariants.
        registerInvariant(invariantAddPresent);
    }

    zen::log("----------------------------");
    zen::log("Starting reduction passes...");
    zen::log("----------------------------");

    // Run registered passes iteratively until no changes occur.
    int pass_count = 0;
    bool pass_applied;
    do { // Apply each pass in the registry.
        pass_applied = false;
        for (auto& pass : getPassRegistry()) {
            // Backup the current state of the module.
            IrModule* backup = cloneModule(module);
            if (pass(module)) {
                // After applying the pass, verify all registered invariants.
                bool invariant_ok = true;
                for (auto& inv : getInvariantRegistry()) {
                    if (!inv(module)) {
                        invariant_ok = false;
                        break;
                    }
                }
                if (!invariant_ok) {
                    zen::log(zen::color::yellow("An invariant failed after the most recent pass; reverting it..."));
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

    zen::log("----------------------------");
    zen::log("Reduction passes ended.");
    zen::log("----------------------------");
    
    std::string final_ir = to_string(module);

    zen::log("Final module after", pass_count, "reductions:\n");
    zen::log(final_ir);                 // still print

    std::filesystem::create_directories(std::filesystem::path(output_file_path).parent_path());
    std::ofstream out_ir(output_file_path);
    if (!out_ir)
        zen::log(zen::color::red("ERROR:"), "Cannot open ", zen::quote(output_file_path), " for writing.");
    else{ 
        out_ir << final_ir << std::endl,
        zen::log("Wrote reduced IR to:", zen::quote(output_file_path));
    }
        
    // Cleanup: deallocate memory. This will be rewritten later
    // with proper resource management after this POC phase.
    freeModule(module);
} catch (const std::exception& e) {
    zen::log(zen::color::red("ERROR:"), e.what());
    return 1;
}
