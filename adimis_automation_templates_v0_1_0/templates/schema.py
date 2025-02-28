import os
import importlib.util

# The top-level package name of your installed package.
PACKAGE_NAME = "adimis_automation_templates_v0_1_0"
# Set the templates folder relative to this file's location.
TEMPLATES_FOLDER = os.path.dirname(__file__)
EXCLUDE_FOLDERS = ["lib"]

def get_graph_templates():
    graph_templates = []
    # Compute the package root (assumed to be one level above the templates folder)
    package_root = os.path.abspath(os.path.join(TEMPLATES_FOLDER, ".."))

    for root, dirs, files in os.walk(TEMPLATES_FOLDER):
        # Exclude folders as needed.
        dirs[:] = [d for d in dirs if d not in EXCLUDE_FOLDERS and not d.startswith('__')]
        if "schema.py" in files:
            schema_path = os.path.join(root, "schema.py")
            # Compute the relative path from the package root.
            rel_path = os.path.relpath(root, package_root)
            # Construct the full package name.
            module_package = PACKAGE_NAME + "." + rel_path.replace(os.sep, ".")
            # Create a unique module name.
            module_name = module_package + "_schema"
            spec = importlib.util.spec_from_file_location(module_name, schema_path,
                                                          submodule_search_locations=[root])
            module = importlib.util.module_from_spec(spec)
            try:
                # Set the module's package to the full package path.
                module.__package__ = module_package
                spec.loader.exec_module(module)
            except Exception as e:
                continue
            if hasattr(module, "graph_schema"):
                graph_templates.append(module.graph_schema)

    return graph_templates