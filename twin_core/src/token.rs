pub fn count_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

// Tests for count_tokens
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn single_word() {
        assert_eq!(count_tokens("hello"), 1);
    }

    #[test]
    fn matches_python_behavior() {
        // Verify parity with Python's len(text.split())
        assert_eq!(count_tokens("hello world foo bar"), 4);
    }
}