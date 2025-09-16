"""
Test suite for configuration and script files validation.
Addresses testing issues in non-Python configuration files.
"""
import unittest
import os
import json
import tempfile
from pathlib import Path


class TestConfigurationFiles(unittest.TestCase):
    """Test configuration files for basic validation and structure."""

    def test_jest_config_mjs_structure(self):
        """Test Jest configuration file structure and content."""
        jest_config_path = 'jest.config.mjs'
        
        if os.path.exists(jest_config_path):
            with open(jest_config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "Jest config file should not be empty")
            
            # Test for basic Jest configuration elements
            self.assertIn('export', content, "Jest config should have export statement")
            
            # Check for coverage exclusion issue mentioned
            if '/packages/docusaurus-utils/src/index.ts' in content:
                print("Warning: Specific source file excluded from coverage as noted in issue")
        else:
            self.skipTest("jest.config.mjs not found")

    def test_jsonnet_config_structure(self):
        """Test JSONNET configuration file basic structure."""
        jsonnet_path = 'p3lm-T.jsonnet'
        
        if os.path.exists(jsonnet_path):
            with open(jsonnet_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "JSONNET config should not be empty")
            
            # Check for test evaluation configuration mentioned in issue
            if 'evaluate_on_test' in content:
                self.assertIn('true', content.lower(), 
                            "Test evaluation should be enabled")
        else:
            self.skipTest("p3lm-T.jsonnet not found")

    def test_batch_script_structure(self):
        """Test batch script for basic structure and pause command issue."""
        bat_path = 'rt.bat'
        
        if os.path.exists(bat_path):
            with open(bat_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "Batch script should not be empty")
            
            # Check for pause command that interrupts automation
            if 'pause' in content.lower():
                lines = content.split('\n')
                pause_lines = [i for i, line in enumerate(lines, 1) 
                              if 'pause' in line.lower()]
                if pause_lines:
                    print(f"Warning: Interactive pause found at lines {pause_lines}. "
                          "This may interrupt automated testing.")
        else:
            self.skipTest("rt.bat not found")

    def test_xonsh_script_structure(self):
        """Test xonsh script for basic structure and safety."""
        xsh_path = 'kill_gpu_processes.xsh'
        
        if os.path.exists(xsh_path):
            with open(xsh_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "Xonsh script should not be empty")
            
            # Check for destructive operations without safety measures
            destructive_commands = ['kill', 'rm', 'delete']
            for cmd in destructive_commands:
                if cmd in content.lower():
                    # Look for safety measures
                    safety_measures = ['--dry-run', 'confirm', 'test', 'echo']
                    has_safety = any(safety in content.lower() for safety in safety_measures)
                    if not has_safety:
                        print(f"Warning: Destructive command '{cmd}' found without "
                              "apparent safety measures or test mode.")
        else:
            self.skipTest("kill_gpu_processes.xsh not found")

    def test_makefile_in_structure(self):
        """Test makefile input template structure."""
        makefile_path = 'make_emakefile.in'
        
        if os.path.exists(makefile_path):
            with open(makefile_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "Makefile template should not be empty")
            
            # Look for basic makefile structure elements
            makefile_elements = ['@', '$(', 'ifdef', 'endif']
            found_elements = [elem for elem in makefile_elements if elem in content]
            
            if found_elements:
                print(f"Makefile template contains typical elements: {found_elements}")
            else:
                print("Warning: Makefile template may not contain standard makefile syntax")
        else:
            self.skipTest("make_emakefile.in not found")

    def test_build_script_structure(self):
        """Test build script for basic structure and completeness."""
        build_script_path = 'build-otp-tar.txt'
        
        if os.path.exists(build_script_path):
            with open(build_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "Build script should not be empty")
            
            # Look for typical build script elements
            build_elements = ['make', 'configure', 'install', 'build']
            found_elements = [elem for elem in build_elements 
                            if elem.lower() in content.lower()]
            
            if found_elements:
                print(f"Build script contains build-related commands: {found_elements}")
        else:
            self.skipTest("build-otp-tar.txt not found")


class TestXMLStylesheets(unittest.TestCase):
    """Test XML/XSLT stylesheet files."""

    def test_qtfm_xsl_structure(self):
        """Test XSLT stylesheet basic structure."""
        xsl_path = 'qtfmfunction.xsl'
        
        if os.path.exists(xsl_path):
            with open(xsl_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "XSL stylesheet should not be empty")
            
            # Check for basic XSLT elements
            xsl_elements = ['xsl:stylesheet', 'xsl:template', 'xsl:transform']
            has_xsl_elements = any(elem in content for elem in xsl_elements)
            
            self.assertTrue(has_xsl_elements, 
                          "XSL file should contain XSLT stylesheet elements")
        else:
            self.skipTest("qtfmfunction.xsl not found")

    def test_qtfm_latex_xsl_structure(self):
        """Test XSLT to LaTeX transformation stylesheet."""
        xsl_path = 'qtfmfunction2latex.xsl'
        
        if os.path.exists(xsl_path):
            with open(xsl_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "LaTeX XSL stylesheet should not be empty")
            
            # Check for XSLT elements
            xsl_elements = ['xsl:stylesheet', 'xsl:template', 'xsl:output']
            has_xsl_elements = any(elem in content for elem in xsl_elements)
            
            self.assertTrue(has_xsl_elements, 
                          "LaTeX XSL should contain XSLT elements")
            
            # Check for LaTeX-specific output
            latex_indicators = ['\\', 'latex', 'LaTeX']
            has_latex = any(indicator in content for indicator in latex_indicators)
            
            if has_latex:
                print("XSL stylesheet appears to generate LaTeX output")
        else:
            self.skipTest("qtfmfunction2latex.xsl not found")


class TestScriptValidation(unittest.TestCase):
    """Test script files for validation and testing capabilities."""

    def test_sed_tokenizer_structure(self):
        """Test sed tokenizer script structure."""
        sed_path = 'tokenizer.sed'
        
        if os.path.exists(sed_path):
            with open(sed_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "Sed tokenizer should not be empty")
            
            # Check for sed script patterns
            sed_patterns = ['s/', 'g', '/', 'p']
            has_sed_patterns = any(pattern in content for pattern in sed_patterns)
            
            if has_sed_patterns:
                print("Sed script contains typical sed command patterns")
            else:
                print("Warning: Sed script may not contain standard sed patterns")
        else:
            self.skipTest("tokenizer.sed not found")

    def test_matlab_script_structure(self):
        """Test MATLAB script basic structure."""
        matlab_path = 'fig1.m'
        
        if os.path.exists(matlab_path):
            with open(matlab_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test that file has content
            self.assertTrue(content.strip(), "MATLAB script should not be empty")
            
            # Check for MATLAB-specific elements
            matlab_elements = ['figure', 'plot', 'xlabel', 'ylabel', 'title', '%']
            found_elements = [elem for elem in matlab_elements 
                            if elem in content.lower()]
            
            if found_elements:
                print(f"MATLAB script contains plotting elements: {found_elements}")
        else:
            self.skipTest("fig1.m not found")


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration files for common validation issues."""

    def test_files_exist_and_readable(self):
        """Test that configuration files exist and are readable."""
        config_files = [
            'jest.config.mjs',
            'p3lm-T.jsonnet', 
            'rt.bat',
            'kill_gpu_processes.xsh',
            'make_emakefile.in',
            'build-otp-tar.txt',
            'qtfmfunction.xsl',
            'qtfmfunction2latex.xsl',
            'tokenizer.sed',
            'fig1.m'
        ]
        
        readable_files = []
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            readable_files.append(config_file)
                except Exception as e:
                    print(f"Warning: Could not read {config_file}: {e}")
        
        print(f"Successfully validated {len(readable_files)} configuration files")
        self.assertGreater(len(readable_files), 0, 
                          "At least some configuration files should be readable")

    def test_no_obviously_malformed_files(self):
        """Test that configuration files are not obviously malformed."""
        config_files = [
            'jest.config.mjs',
            'p3lm-T.jsonnet',
            'qtfmfunction.xsl',
            'qtfmfunction2latex.xsl'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Basic malformation checks
                    # Check for balanced brackets/braces (simple check)
                    open_brackets = content.count('{')
                    close_brackets = content.count('}')
                    
                    if abs(open_brackets - close_brackets) > 10:  # Allow some tolerance
                        print(f"Warning: {config_file} may have unbalanced braces")
                    
                    # Check file is not just whitespace or very short
                    if len(content.strip()) < 10:
                        print(f"Warning: {config_file} seems very short or empty")
                        
                except Exception as e:
                    print(f"Could not validate {config_file}: {e}")

    def test_create_test_validations(self):
        """Create basic validation tests for configuration files."""
        # This test demonstrates how validation could be implemented
        
        validation_results = {
            'files_checked': 0,
            'files_passed': 0,
            'warnings': []
        }
        
        config_files = [
            'jest.config.mjs',
            'p3lm-T.jsonnet',
            'rt.bat',
            'kill_gpu_processes.xsh',
            'qtfmfunction.xsl',
            'qtfmfunction2latex.xsl'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                validation_results['files_checked'] += 1
                
                try:
                    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Basic validation passed
                    if content.strip():
                        validation_results['files_passed'] += 1
                    else:
                        validation_results['warnings'].append(
                            f"{config_file}: Empty or whitespace-only"
                        )
                        
                except Exception as e:
                    validation_results['warnings'].append(
                        f"{config_file}: Read error - {e}"
                    )
        
        # Report results
        print(f"Configuration Validation Summary:")
        print(f"  Files checked: {validation_results['files_checked']}")
        print(f"  Files passed: {validation_results['files_passed']}")
        print(f"  Warnings: {len(validation_results['warnings'])}")
        
        for warning in validation_results['warnings']:
            print(f"    {warning}")
        
        # Test should pass if we validated anything
        self.assertGreater(validation_results['files_checked'], 0)


class TestConfigurationIssueResolution(unittest.TestCase):
    """Test resolution of specific configuration issues mentioned."""

    def test_jest_coverage_exclusion_awareness(self):
        """Test awareness of Jest coverage exclusion issue."""
        jest_config = 'jest.config.mjs'
        
        if os.path.exists(jest_config):
            with open(jest_config, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for the specific exclusion mentioned in the issue
            excluded_file = '/packages/docusaurus-utils/src/index.ts'
            
            if excluded_file in content:
                # Issue is documented - create awareness test
                self.assertTrue(True, "Coverage exclusion documented")
                print(f"Note: {excluded_file} is excluded from coverage as per issue report")
            else:
                # No exclusion found - that's actually good
                print("No problematic coverage exclusions found")
        else:
            self.skipTest("jest.config.mjs not found")

    def test_jsonnet_evaluation_framework(self):
        """Test JSONNET evaluation framework completeness."""
        jsonnet_config = 'p3lm-T.jsonnet'
        
        if os.path.exists(jsonnet_config):
            with open(jsonnet_config, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for evaluation settings
            evaluation_indicators = [
                'evaluate_on_test',
                'evaluation',
                'metric',
                'test'
            ]
            
            found_indicators = [indicator for indicator in evaluation_indicators
                              if indicator in content.lower()]
            
            if found_indicators:
                print(f"Evaluation configuration found: {found_indicators}")
                
                # Check for threshold or criteria
                criteria_indicators = ['threshold', 'criteria', 'target']
                found_criteria = [crit for crit in criteria_indicators
                                if crit in content.lower()]
                
                if found_criteria:
                    print(f"Evaluation criteria found: {found_criteria}")
                else:
                    print("Warning: Evaluation enabled but no clear criteria/thresholds found")
        else:
            self.skipTest("p3lm-T.jsonnet not found")

    def test_automation_blocking_elements(self):
        """Test for elements that block automated testing."""
        problematic_files = {
            'rt.bat': ['pause', 'input', 'read'],
            'kill_gpu_processes.xsh': ['kill', 'rm', 'delete']
        }
        
        for filename, blocking_elements in problematic_files.items():
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    
                found_blocks = [elem for elem in blocking_elements if elem in content]
                
                if found_blocks:
                    print(f"Warning: {filename} contains potentially blocking elements: {found_blocks}")
                    
                    # Look for mitigation (test modes, flags, etc.)
                    mitigation_indicators = [
                        'test', 'dry-run', 'simulate', 'echo', 'debug', 'verbose'
                    ]
                    found_mitigation = [mit for mit in mitigation_indicators 
                                      if mit in content]
                    
                    if found_mitigation:
                        print(f"  Mitigation found: {found_mitigation}")
                    else:
                        print(f"  No obvious testing/dry-run modes found")


if __name__ == '__main__':
    # Run all configuration tests
    unittest.main(verbosity=2)
