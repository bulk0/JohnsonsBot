# Test Data Guide

This directory contains various SPSS test files used for testing the Johnson's Bot functionality. The files are organized into different categories based on their testing purpose.

## Directory Structure

```
test_data/
├── scenarios/     # Standard use-case scenario files
├── edge_cases/    # Files testing boundary conditions
├── error_cases/   # Files for error handling testing
└── performance/   # Files for performance testing
```

## Current Test Files

### Scenarios Directory
Currently contains our standard scenario test files:
- Scen1-B_aggr.sav
- Scen2-C_aggr.sav
- Scen3-D_aggr.sav
- Scen4-E_aggr.sav
- Scen5-F_aggr.sav
- Scen6-G_aggr.sav
- Scen7-H_aggr.sav
- Scen8-I_aggr.sav
- Scen9-J_aggr.sav
- Scen10-K_aggr.sav
- Scen11-L_aggr.sav
- Scen12-M_aggr.sav

## Test Categories

### 1. Scenarios (scenarios/)
- Purpose: Test standard use-cases and expected behavior
- What to test:
  - Normal data processing flows
  - Different variable combinations
  - Various brand scenarios
  - Different analysis scopes

### 2. Edge Cases (edge_cases/)
- Purpose: Test boundary conditions and special cases
- What to test:
  - Maximum/minimum number of variables
  - Special characters in variable names
  - Different encoding types
  - Missing data scenarios
  - Extreme value cases
  - Different SPSS file versions

Current test files:
1. Yandex_CSI 2025 Poisk Finsrez_(FINAL_for_client)_v2.sav
   - Purpose: Test handling of complex file characteristics
   - Characteristics:
     * Non-standard file name with spaces and special characters
     * Cyrillic character encoding (non-UTF-8)
     * Client-specific naming conventions
   - Expected behavior:
     * Should handle file names with spaces correctly
     * Should detect and handle non-UTF-8 encoding
     * Should properly escape special characters in file paths
     * Should validate file structure despite encoding challenges

### 3. Error Cases (error_cases/)
- Purpose: Test error handling and recovery
- What to test:
  - Malformed SPSS files
  - Invalid data structures
  - Corrupted files
  - Invalid variable types
  - Incorrect variable relationships
  - Missing required fields

Current test files:
1. base.sav
   - Purpose: Test handling of non-standard survey data structure
   - Characteristics:
     * Large number of variables (116 columns)
     * Mix of data types (object and float64)
     * Non-standard variable naming (q1_1, q1_2, etc.)
     * Survey metadata columns (timeStart, timeFinish, browser, etc.)
     * Special codes (_98 suffix variables indicating "Other" options)
   - Expected behavior:
     * Should identify numeric vs non-numeric variables correctly
     * Should handle special survey codes (98, 99) appropriately
     * Should validate variable relationships despite non-standard naming

### 4. Performance (performance/)
- Purpose: Test system performance and resource handling
- What to test:
  - Large datasets
  - Multiple concurrent processing
  - Memory usage scenarios
  - Processing time benchmarks

## Adding New Test Files

When adding new test files:
1. Place them in the appropriate category directory
2. Update this guide with:
   - File name
   - Purpose/scenario being tested
   - Expected behavior
   - Any special considerations

## Test File Naming Convention

For new test files, use the following naming convention:
- Scenario files: `ScenX-[ID]_[type].sav`
- Edge cases: `edge_[scenario]_[type].sav`
- Error cases: `error_[type]_[scenario].sav`
- Performance: `perf_[size]_[scenario].sav`

## Test Results

Test results are stored in the `test_results/` directory at the project root level.
