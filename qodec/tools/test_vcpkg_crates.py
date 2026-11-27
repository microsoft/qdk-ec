import unittest

import vcpkg_crates


def package(name: str, source: str | None = vcpkg_crates.CRATES_IO) -> dict:
    return {"id": name, "name": name, "version": "1.0.0", "source": source}


def edge(target: str, *kinds: str | None) -> dict:
    return {"pkg": target, "dep_kinds": [{"kind": kind} for kind in kinds]}


def metadata(packages: list[dict], edges: dict[str, list[dict]]) -> dict:
    nodes = [{"id": p["id"], "deps": edges.get(p["id"], [])} for p in packages]
    return {"packages": packages, "resolve": {"nodes": nodes}}


class BuildClosureTests(unittest.TestCase):
    def test_follows_normal_and_build_edges_but_not_dev_only_edges(self):
        packages = [package("qodec-c", None), package("qodec", None), package("serde"),
                    package("serde_derive"), package("tempfile"), package("syn")]
        edges = {
            "qodec-c": [edge("qodec", None), edge("tempfile", "dev")],
            "qodec": [edge("serde", None, "dev"), edge("serde_derive", "build")],
            "serde_derive": [edge("syn", None)],
        }
        closure = vcpkg_crates.build_closure(metadata(packages, edges), "qodec-c")
        self.assertEqual([p["name"] for p in closure], ["serde", "serde_derive", "syn"])

    def test_rejects_a_dependency_that_cannot_come_from_crates_io(self):
        packages = [package("qodec-c", None), package("forked", "git+https://example.com/forked")]
        with self.assertRaisesRegex(ValueError, "forked"):
            vcpkg_crates.build_closure(metadata(packages, {"qodec-c": [edge("forked", None)]}), "qodec-c")


if __name__ == "__main__":
    unittest.main()
