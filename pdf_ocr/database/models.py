from database.db_setup import Base

from sqlalchemy import Column, Integer, String, Date, UniqueConstraint

class DataPoint(Base):
    __tablename__ = "data_points"

    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String, index=True)
    filing_number = Column(String, index=True)
    filing_date = Column(Date)
    rcs_number = Column(String)
    dp_value = Column(String)
    dp_unique_value = Column(String)
    page_number = Column(Integer)  # Add page_number field

    __table_args__ = (UniqueConstraint('unique_id', 'dp_unique_value', 'page_number', name='_unique_data_point'),)
