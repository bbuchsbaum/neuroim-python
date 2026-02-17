#!/usr/bin/env python3
"""
Golden Test Synchronization Tool

Synchronizes golden test specifications between source and target repositories
while preserving language-specific implementations.

Usage:
    python sync_golden_tests.py --source PATH --target PATH --language LANG [options]

Example:
    python sync_golden_tests.py \
        --source ~/code/neuroim2/golden_tests \
        --target ./golden_tests \
        --language Python
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import shutil
from datetime import datetime
import difflib


class GoldenTestSync:
    """Synchronizes golden tests between repositories."""
    
    def __init__(self, source_dir: str, target_dir: str, language: str, 
                 dry_run: bool = False, verbose: bool = False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.language = language
        self.dry_run = dry_run
        self.verbose = verbose
        self.namespace = {'gt': 'http://golden-tests.org/schema'}
        
        # Track changes
        self.new_tests: List[str] = []
        self.modified_tests: List[str] = []
        self.deleted_tests: List[str] = []
        self.implemented_tests: List[str] = []
        self.unimplemented_tests: List[str] = []
    
    def sync(self):
        """Main synchronization method."""
        print(f"Golden Test Sync Report")
        print(f"======================")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print(f"Language: {self.language}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print()
        
        # Validate directories
        if not self.source_dir.exists():
            print(f"ERROR: Source directory does not exist: {self.source_dir}")
            return False
        
        if not self.target_dir.exists():
            print(f"ERROR: Target directory does not exist: {self.target_dir}")
            return False
        
        # Find all test files
        source_tests = self._find_xml_files(self.source_dir)
        target_tests = self._find_xml_files(self.target_dir)
        
        if self.verbose:
            print(f"Found {len(source_tests)} tests in source")
            print(f"Found {len(target_tests)} tests in target")
            print()
        
        # Process tests
        self._identify_changes(source_tests, target_tests)
        self._sync_tests(source_tests, target_tests)
        self._report_summary()
        
        return True
    
    def _find_xml_files(self, base_dir: Path) -> Dict[str, Path]:
        """Find all XML files relative to base directory."""
        xml_files = {}
        specs_dir = base_dir / "specs"
        
        if specs_dir.exists():
            for xml_path in specs_dir.rglob("*.xml"):
                rel_path = xml_path.relative_to(base_dir)
                xml_files[str(rel_path)] = xml_path
        
        return xml_files
    
    def _identify_changes(self, source_tests: Dict[str, Path], 
                         target_tests: Dict[str, Path]):
        """Identify new, modified, and deleted tests."""
        source_keys = set(source_tests.keys())
        target_keys = set(target_tests.keys())
        
        # New tests
        self.new_tests = sorted(source_keys - target_keys)
        
        # Deleted tests  
        self.deleted_tests = sorted(target_keys - source_keys)
        
        # Check for modifications
        common_tests = source_keys & target_keys
        for test_path in common_tests:
            source_spec = self._extract_test_spec(source_tests[test_path])
            target_spec = self._extract_test_spec(target_tests[test_path])
            
            if source_spec != target_spec:
                self.modified_tests.append(test_path)
        
        self.modified_tests.sort()
    
    def _extract_test_spec(self, xml_path: Path) -> str:
        """Extract test specification (excluding implementations)."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Remove implementation sections to compare only specs
        implementations = root.find('.//gt:implementations', self.namespace)
        if implementations is not None:
            root.remove(implementations)
        
        # Remove propagation status as it changes with implementations
        prop_status = root.find('.//gt:propagation_status', self.namespace)
        if prop_status is not None:
            root.remove(prop_status)
        
        # Convert to string for comparison
        return ET.tostring(root, encoding='unicode', method='xml')
    
    def _extract_implementation(self, xml_path: Path, language: str) -> Optional[str]:
        """Extract language-specific implementation from XML."""
        tree = ET.parse(xml_path)
        impl = tree.find(f'.//gt:{language}', self.namespace)
        
        if impl is not None:
            return impl.text
        return None
    
    def _sync_tests(self, source_tests: Dict[str, Path], 
                   target_tests: Dict[str, Path]):
        """Synchronize tests while preserving implementations."""
        
        # Process new tests
        for test_path in self.new_tests:
            self._copy_new_test(source_tests[test_path], test_path)
        
        # Process modified tests
        for test_path in self.modified_tests:
            self._update_test(source_tests[test_path], target_tests[test_path])
        
        # Check implementation status
        all_tests = set(source_tests.keys()) | set(target_tests.keys())
        for test_path in all_tests:
            if test_path in self.deleted_tests:
                continue
                
            target_path = target_tests.get(test_path)
            if not target_path:
                target_path = self.target_dir / test_path
                
            if target_path.exists():
                impl = self._extract_implementation(target_path, self.language)
                if impl and impl.strip():
                    self.implemented_tests.append(test_path)
                else:
                    self.unimplemented_tests.append(test_path)
    
    def _copy_new_test(self, source_path: Path, rel_path: str):
        """Copy a new test from source to target."""
        target_path = self.target_dir / rel_path
        
        if self.verbose:
            print(f"NEW: {rel_path}")
        
        if not self.dry_run:
            # Create target directory if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, target_path)
            
            if self.verbose:
                print(f"  -> Copied to {target_path}")
    
    def _update_test(self, source_path: Path, target_path: Path):
        """Update test specification while preserving implementation."""
        if self.verbose:
            print(f"MODIFIED: {target_path.relative_to(self.target_dir)}")
        
        # Extract current implementation
        current_impl = self._extract_implementation(target_path, self.language)
        
        if not self.dry_run:
            # Parse source XML
            source_tree = ET.parse(source_path)
            source_root = source_tree.getroot()
            
            # If we have an implementation, preserve it
            if current_impl is not None:
                # Find or create implementations section
                impl_section = source_root.find('.//gt:implementations', self.namespace)
                if impl_section is None:
                    impl_section = ET.SubElement(source_root, 'implementations')
                
                # Find or create language element
                lang_elem = impl_section.find(f'gt:{self.language}', self.namespace)
                if lang_elem is None:
                    lang_elem = ET.SubElement(impl_section, self.language)
                
                lang_elem.text = current_impl
                
                # Update propagation status
                self._update_propagation_status(source_root, self.language)
            
            # Write updated XML
            self._write_xml(source_tree, target_path)
            
            if self.verbose:
                print(f"  -> Updated (preserved {self.language} implementation)")
    
    def _update_propagation_status(self, root: ET.Element, language: str):
        """Update propagation status for the language."""
        prop_section = root.find('.//gt:propagation_status', self.namespace)
        if prop_section is None:
            prop_section = ET.SubElement(root, 'propagation_status')
        
        # Find existing status for this language
        lang_status = None
        for impl in prop_section.findall('gt:implementation', self.namespace):
            if impl.get('lang') == language:
                lang_status = impl
                break
        
        # Update or create status
        if lang_status is None:
            lang_status = ET.SubElement(prop_section, 'implementation')
            lang_status.set('lang', language)
        
        lang_status.set('status', 'complete')
        lang_status.set('date', datetime.now().strftime('%Y-%m-%d'))
    
    def _write_xml(self, tree: ET.ElementTree, path: Path):
        """Write XML with proper formatting."""
        # Add indentation for readability
        self._indent_xml(tree.getroot())
        
        # Write with XML declaration
        tree.write(path, encoding='UTF-8', xml_declaration=True, method='xml')
    
    def _indent_xml(self, elem: ET.Element, level: int = 0):
        """Add indentation to XML for readability."""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
    
    def _report_summary(self):
        """Print summary report."""
        print(f"\nSummary:")
        print(f"--------")
        
        if self.new_tests:
            print(f"New tests found: {len(self.new_tests)}")
            for test in self.new_tests[:5]:  # Show first 5
                print(f"  - {test}")
            if len(self.new_tests) > 5:
                print(f"  ... and {len(self.new_tests) - 5} more")
        
        if self.modified_tests:
            print(f"\nModified tests: {len(self.modified_tests)}")
            for test in self.modified_tests[:5]:
                print(f"  - {test}")
            if len(self.modified_tests) > 5:
                print(f"  ... and {len(self.modified_tests) - 5} more")
        
        if self.deleted_tests:
            print(f"\nDeleted tests: {len(self.deleted_tests)}")
            for test in self.deleted_tests[:5]:
                print(f"  - {test}")
            if len(self.deleted_tests) > 5:
                print(f"  ... and {len(self.deleted_tests) - 5} more")
        
        # Implementation status
        total_tests = len(self.implemented_tests) + len(self.unimplemented_tests)
        if total_tests > 0:
            print(f"\nImplementation Status:")
            print(f"  Total tests: {total_tests}")
            print(f"  [✓] Implemented: {len(self.implemented_tests)}")
            print(f"  [✗] Need implementation: {len(self.unimplemented_tests)}")
            
            if self.unimplemented_tests and self.verbose:
                print(f"\nTests needing {self.language} implementation:")
                for test in sorted(self.unimplemented_tests)[:10]:
                    print(f"  - {test}")
                if len(self.unimplemented_tests) > 10:
                    print(f"  ... and {len(self.unimplemented_tests) - 10} more")
        
        if self.dry_run:
            print(f"\nDRY RUN: No changes were made. Remove --dry-run to apply changes.")
        
        print(f"\nRun with --verbose for detailed information.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize golden tests between repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sync
  %(prog)s --source ~/code/neuroim2/golden_tests --target ./golden_tests --language Python

  # Dry run with verbose output  
  %(prog)s --source ~/code/neuroim2/golden_tests --target ./golden_tests --language Python --dry-run --verbose
        """
    )
    
    parser.add_argument('--source', '-s', required=True,
                        help='Source repository golden tests directory')
    parser.add_argument('--target', '-t', required=True,
                        help='Target repository golden tests directory')
    parser.add_argument('--language', '-l', required=True,
                        help='Target language implementations to preserve (e.g., Python, Rust, Julia)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview changes without applying them')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    # Run sync
    syncer = GoldenTestSync(
        source_dir=args.source,
        target_dir=args.target,
        language=args.language,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    success = syncer.sync()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()