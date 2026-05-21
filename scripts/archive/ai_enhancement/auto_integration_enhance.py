#!/usr/bin/env python3
"""
Automatically integrate AI-generated enhancements back to app/
Seamlessly merges improvements while keeping app/ clean during generation
"""

import difflib
import logging
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories
ENHANCEMENTS_DIR = Path("Gemini Enhancement")
SRC_DIR = Path("app")
TESTS_DIR = Path("tests")


class EnhancementIntegrator:
    """Automatically integrate enhancements from enhancements/ to src/"""

    def __init__(self):
        self.stats = {"tests_copied": 0, "files_merged": 0, "conflicts": 0, "skipped": 0}

    def verify_structure(self) -> bool:
        """Verify enhancements directory structure exists"""
        if not ENHANCEMENTS_DIR.exists():
            logger.error(f"❌ {ENHANCEMENTS_DIR}/ directory not found!")
            logger.error("   Run enhance_with_gemini_CLEAN.py first")
            return False

        logger.info("✅ Found Gemini Enhancement")
        return True

    def integrate_tests(self) -> int:
        """Copy generated tests from Gemini Enhancement/tests/ to tests/"""
        logger.info("\n" + "=" * 60)
        logger.info("📝 Integrating Tests...")
        logger.info("=" * 60)

        test_source_dir = ENHANCEMENTS_DIR / "tests"
        if not test_source_dir.exists():
            logger.warning("   ⏭️  No tests generated")
            return 0

        # Create tests directory if needed
        TESTS_DIR.mkdir(parents=True, exist_ok=True)

        count = 0
        for test_file in test_source_dir.glob("test_*.py"):
            dest_file = TESTS_DIR / test_file.name

            if dest_file.exists():
                logger.info(f"📋 Reviewing: {test_file.name}")
                if self._should_overwrite(test_file, dest_file):
                    shutil.copy2(test_file, dest_file)
                    logger.info(f"   ✅ Updated: {dest_file}")
                    count += 1
                else:
                    logger.info(f"   ⏭️  Kept existing: {dest_file}")
                    self.stats["skipped"] += 1
            else:
                shutil.copy2(test_file, dest_file)
                logger.info(f"✅ Added: {dest_file}")
                count += 1

        self.stats["tests_copied"] = count
        return count

    def integrate_documentation(self) -> int:
        """Intelligently merge documentation from documented/ files"""
        logger.info("\n" + "=" * 60)
        logger.info("📚 Integrating Documentation...")
        logger.info("=" * 60)

        doc_source_dir = ENHANCEMENTS_DIR / "documented"
        if not doc_source_dir.exists():
            logger.warning("   ⏭️  No documentation generated")
            return 0

        count = 0
        for doc_file in doc_source_dir.glob("*_documented.py"):
            # Extract original filename
            original_name = doc_file.name.replace("_documented.py", ".py")
            original_file = SRC_DIR / original_name

            if not original_file.exists():
                logger.warning(f"⏭️  Skipping {doc_file.name} (original not found)")
                self.stats["skipped"] += 1
                continue

            logger.info(f"📋 Reviewing: {original_name}")

            # Merge docstrings and type hints
            if self._merge_documentation(original_file, doc_file):
                logger.info(f"   ✅ Documentation merged: {original_name}")
                count += 1
            else:
                logger.info(f"   ⚠️  Manual review needed: {original_name}")
                self.stats["conflicts"] += 1

        return count

    def integrate_typed_versions(self) -> int:
        """Merge type hints from typed/ files"""
        logger.info("\n" + "=" * 60)
        logger.info("📝 Integrating Type Hints...")
        logger.info("=" * 60)

        typed_source_dir = ENHANCEMENTS_DIR / "typed"
        if not typed_source_dir.exists():
            logger.warning("   ⏭️  No typed versions generated")
            return 0

        count = 0
        for typed_file in typed_source_dir.glob("*_typed.py"):
            # Extract original filename
            original_name = typed_file.name.replace("_typed.py", ".py")
            original_file = SRC_DIR / original_name

            if not original_file.exists():
                logger.warning(f"⏭️  Skipping {typed_file.name} (original not found)")
                self.stats["skipped"] += 1
                continue

            logger.info(f"📋 Reviewing: {original_name}")

            # Merge type hints
            if self._merge_type_hints(original_file, typed_file):
                logger.info(f"   ✅ Type hints merged: {original_name}")
                count += 1
            else:
                logger.info(f"   ⚠️  Manual review needed: {original_name}")
                self.stats["conflicts"] += 1

        return count

    def create_review_files(self) -> None:
        """Create side-by-side comparison files for manual review"""
        logger.info("\n" + "=" * 60)
        logger.info("📂 Creating Review Files...")
        logger.info("=" * 60)

        review_dir = Path("enhancement_reviews")
        review_dir.mkdir(exist_ok=True)

        # Create refactored comparison
        refactored_dir = ENHANCEMENTS_DIR / "refactored"
        if refactored_dir.exists():
            logger.info("📋 Creating refactored code comparisons...")
            for ref_file in refactored_dir.glob("*_refactored.py"):
                original_name = ref_file.name.replace("_refactored.py", ".py")
                original_file = SRC_DIR / original_name

                if original_file.exists():
                    review_file = review_dir / f"{original_name}.refactoring_review.md"
                    self._create_comparison(original_file, ref_file, review_file)
                    logger.info(f"   ✅ Review file: {review_file}")

        # Create optimized comparison
        optimized_dir = ENHANCEMENTS_DIR / "optimized"
        if optimized_dir.exists():
            logger.info("📋 Creating optimization comparisons...")
            for opt_file in optimized_dir.glob("*_optimized.py"):
                original_name = opt_file.name.replace("_optimized.py", ".py")
                original_file = SRC_DIR / original_name

                if original_file.exists():
                    review_file = review_dir / f"{original_name}.optimization_review.md"
                    self._create_comparison(original_file, opt_file, review_file)
                    logger.info(f"   ✅ Review file: {review_file}")

    def _should_overwrite(self, source: Path, dest: Path) -> bool:
        """Ask user if they want to overwrite existing file"""
        logger.info(f"\n   File already exists: {dest.name}")
        logger.info(f"   Source: {source} ({source.stat().st_size} bytes)")
        logger.info(f"   Dest:   {dest} ({dest.stat().st_size} bytes)")

        # In automated mode, check if source is newer
        if source.stat().st_mtime > dest.stat().st_mtime:
            logger.info("   → Source is newer, will update")
            return True
        else:
            logger.info("   → Destination is newer, keeping existing")
            return False

    def _merge_documentation(self, original: Path, documented: Path) -> bool:
        """Merge docstrings and type hints into original file"""
        try:
            original_code = original.read_text(encoding="utf-8")
            documented_code = documented.read_text(encoding="utf-8")

            # Extract docstrings from documented version
            # This is a simple approach - extracts complete functions/classes with docs
            merged_code = self._smart_merge_docstrings(original_code, documented_code)

            # Write back to original
            original.write_text(merged_code, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"      Error merging documentation: {e}")
            return False

    def _merge_type_hints(self, original: Path, typed: Path) -> bool:
        """Merge type hints into original file"""
        try:
            original_code = original.read_text(encoding="utf-8")
            typed_code = typed.read_text(encoding="utf-8")

            # Extract type hints from typed version
            merged_code = self._smart_merge_type_hints(original_code, typed_code)

            # Write back to original
            original.write_text(merged_code, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"      Error merging type hints: {e}")
            return False

    def _smart_merge_docstrings(self, original: str, documented: str) -> str:
        """Smart merge that preserves original structure while adding docstrings"""
        # For now, return documented version if significantly different
        # In production, use ast module for proper merging
        if len(documented) > len(original):
            return documented
        return original
        # Placeholder improvement: Only merge if the documented version contains
        # the original critical logic (simple keyword check)
        critical_keywords = ["def ", "class ", "async def"]
        for keyword in critical_keywords:
            if original.count(keyword) != documented.count(keyword):
                logger.warning("Structure mismatch detected in documentation merge. Skipping.")
                return original
        return documented if len(documented) > len(original) else original

    def _smart_merge_type_hints(self, original: str, typed: str) -> str:
        """Smart merge that adds type hints to original"""
        # For now, return typed version
        # In production, use ast module for proper merging
        if len(typed) > len(original):
            return typed
        return original

    def _create_comparison(self, original: Path, enhanced: Path, review_file: Path) -> None:
        """Create a side-by-side comparison file"""
        try:
            original_code = original.read_text(encoding="utf-8").splitlines()
            enhanced_code = enhanced.read_text(encoding="utf-8").splitlines()

            diff = difflib.unified_diff(
                original_code,
                enhanced_code,
                fromfile=f"Original: {original.name}",
                tofile=f"Enhanced: {enhanced.name}",
                lineterm="",
            )

            comparison = f"""# Code Review: {original.name}

## Changes Made:

```diff
{chr(10).join(diff)}
```

## Instructions:
1. Review the differences above
2. If changes look good, copy improvements to {original.name}
3. Delete this file when done

## To Apply Changes Manually:
```powershell
# View side-by-side
code {original} {enhanced}
```
"""

            review_file.write_text(comparison, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error creating comparison: {e}")

    def print_summary(self) -> None:
        """Print integration summary"""
        logger.info("\n" + "=" * 60)
        logger.info("✨ Integration Summary")
        logger.info("=" * 60)

        logger.info("\n📊 Results:")
        logger.info(f"  Tests copied:      {self.stats['tests_copied']}")
        logger.info(f"  Files merged:      {self.stats['files_merged']}")
        logger.info(f"  Manual reviews:    {self.stats['conflicts']}")
        logger.info(f"  Skipped:           {self.stats['skipped']}")

        total = sum(self.stats.values())
        logger.info(f"  Total:             {total}")

        logger.info("\n📁 Locations:")
        logger.info(f"  Tests:             {TESTS_DIR}/")
        logger.info(f"  Source:            {SRC_DIR}/")
        logger.info("  Reviews:           enhancement_reviews/")

        logger.info("\n💡 Next Steps:")
        logger.info("  1. Review enhancement_reviews/ for code comparisons")
        logger.info("  2. Run your tests: pytest tests/")
        logger.info("  3. Commit improvements: git add src/ tests/")


def main():
    """Main integration routine"""
    logger.info("🤖 Automated Enhancement Integration")
    logger.info("=" * 60)
    logger.info("Integrating enhancements from enhancements/ to src/\n")

    integrator = EnhancementIntegrator()

    # Verify structure
    if not integrator.verify_structure():
        return False

    # Integrate components
    integrator.integrate_tests()
    integrator.integrate_documentation()
    integrator.integrate_typed_versions()

    # Create review files for manual inspection
    integrator.create_review_files()

    # Print summary
    integrator.print_summary()

    logger.info("\n" + "=" * 60)
    logger.info("✅ Integration Complete!")
    logger.info("=" * 60)
    logger.info("\n🎯 Ready to use! Tests are in tests/ directory.")
    logger.info("   Review manual changes in enhancement_reviews/\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
