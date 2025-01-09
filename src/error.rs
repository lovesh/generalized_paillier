#[derive(Debug, Clone)]
pub enum PaillierError {
    MessageOutOfBound,
    RandomnessOutOfBound,
    CiphertextOutOfBound,
}
